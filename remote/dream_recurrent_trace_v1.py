"""Frozen QASC cross/self trajectories and common-input functional contrasts.

GPU-only. Author tokenization, forward, aggregation, projection, encoding and
model-loading helpers remain unchanged. Forward hooks only copy label logits.
The 16-item validity stage compares against separate author evaluation calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm GPU allocation required'
    import torch
    import reproduce_qasc as author
    import phase8_qasc_hiddenbench_dense_prefix as original
    from hidden_messages.utils.checksums import sha256_tensor

    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['validity','full'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--validity-receipt',type=Path)
    args=parser.parse_args()
    assert torch.cuda.is_available()
    assert 'L4' in torch.cuda.get_device_name()
    args.out.mkdir(parents=True,exist_ok=False)
    base=Path('vendor/hidden_messages_f85b6d8')
    revision=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()
    assert revision=='f85b6d8563a2236117407e0df9c6cd60ea406dac'
    protocol=json.loads(Path('dream_trace_protocol_v1.json').read_text())
    prior=None
    if args.stage=='full':
        assert args.validity_receipt, 'Full experiment requires completed fixed-cohort trace validation'
        prior=json.loads(args.validity_receipt.read_text())
        assert prior['stage']=='validity' and prior['completed']
        assert prior['numerical_validity_passed'] and prior['public_reference_all_equal']
        assert prior['code_sha256']==sha(__file__)
        assert prior['protocol_sha256']==sha('dream_trace_protocol_v1.json')
        assert prior['revision']==revision and prior['torch']==torch.__version__
    config=author.load_config(base/'configs/qasc_original_evaluation.yaml')
    data=base/'data/qasc/validation.jsonl'
    ckpt=base/'checkpoints/qasc_dense_latent_prefix_channel.pt'
    assert sha(data)==config['qasc']['development_sha256']
    assert sha(ckpt)==config['post_hoc']['source_channel_sha256']
    started=time.monotonic()
    summary=dict(job_id=os.environ['SLURM_JOB_ID'],stage=args.stage,revision=revision,
        code_sha256=sha(__file__),protocol_sha256=sha('dream_trace_protocol_v1.json'),
        data_sha256=sha(data),checkpoint_sha256=sha(ckpt),completed=False,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    if prior:
        summary['prior_validity_receipt_sha256']=sha(args.validity_receipt)
        summary['numerical_validity_scope']='Previously completed fixed16 cohort; not an all-item batch-equivalence claim'
    else:
        summary['numerical_validity_scope']='Separate author calls on this fixed16 cohort'
    (args.out/'started.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=author.load_qasc_rows(data,expected_rows=1000,expected_split='validation')
    if args.stage=='validity':
        lookup={row.example_id:row for row in rows}
        rows=[lookup[key] for key in protocol['validity_example_ids']]
    n=len(rows)
    mc=config['model']
    snapshot=Path('hf_cache/hub')/('models--'+mc['id'].replace('/','--'))/'snapshots'/mc['revision']
    manifest,_=author.model_weight_manifest(snapshot)
    assert manifest==mc['weight_manifest_sha256']
    summary['model_weight_manifest_sha256']=manifest
    author.seed_everything(int(config['training']['seed']))
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model_args=author.dllm.utils.ModelArguments(model_name_or_path=str(snapshot),
        dtype=mc['dtype'],attn_implementation=mc['requested_attention_implementation'])
    model=author.dllm.utils.get_model(model_args=model_args).eval()
    model.requires_grad_(False)
    tokenizer=author.dllm.utils.get_tokenizer(model_args=model_args)
    decoder,embedding,head=model.get_decoder(),model.get_input_embeddings(),model.get_output_embeddings()
    cohort=author.tokenize_cohort(rows,benchmark='qasc',tokenizer=tokenizer,config=config)
    with (args.out/'token_rows.jsonl').open('x') as handle:
        for row,tokens in zip(rows,cohort.token_rows,strict=True):
            handle.write(json.dumps(dict(example_id=row.example_id,agent_token_ids=tokens))+'\n')
    cc=config['channel']
    channel=author.DenseLatentPrefixChannel(int(model.config.hidden_size),
        len(config['surface']['qasc_labels']),input_embedding_rms=author.embedding_rms(embedding),
        message_slots=int(config['surface']['message_slots']),message_dim=int(cc['message_dim']),
        width=int(cc['width']),classifier_width=int(cc['classifier_width'])).to(device)
    channel.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True),strict=True)
    channel.eval().requires_grad_(False)
    diagnostic_batch=int(config['training']['diagnostic_batch_size'])
    label_ids=torch.tensor(cohort.label_ids,device=device)
    common_args=(model,decoder,embedding,head,tokenizer,cohort)
    forward_seconds=[]

    @torch.inference_mode()
    def forward(prefixes=None,receiver_only=False):
        captured=[]
        def capture(_module,_inputs,output):
            # The optimized original path calls the head only on selected states.
            assert output.ndim==2
            captured.append(output.float().index_select(-1,label_ids).cpu())
        hook=head.register_forward_hook(capture)
        torch.cuda.synchronize()
        t=time.monotonic()
        try:
            hidden,predictions=original._forward_cohort(*common_args,
                batch_examples=int(config['evaluation']['qasc_batch_examples']),
                output_position_offset=int(mc['output_position_offset']),prefixes=prefixes,
                receiver_only=receiver_only,reference_full_model=False,device=device)
        finally:
            hook.remove()
        torch.cuda.synchronize()
        forward_seconds.append(time.monotonic()-t)
        logits=torch.cat(captured)
        assert torch.equal(logits.argmax(-1),predictions)
        assert torch.isfinite(logits).all()
        return hidden,predictions,logits

    @torch.inference_mode()
    def classifier(messages):
        flat=messages.reshape(-1,channel.message_slots,channel.message_dim)
        outputs=[]
        for start in range(0,len(flat),diagnostic_batch):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                value=channel.classify_messages(flat[start:start+diagnostic_batch].to(device))
            outputs.append(value.float().cpu())
        return torch.cat(outputs).reshape(n,2,-1)

    @torch.inference_mode()
    def observe(prefixes=None):
        hidden,predictions,logits=forward(prefixes)
        messages=original.encode_all(channel,hidden,batch_size=diagnostic_batch,device=device)
        messages=messages.reshape(n,2,channel.message_slots,channel.message_dim)
        return dict(selected_hidden=hidden.reshape(n,2,-1),predictions=predictions.reshape(n,2),
                    label_logits=logits.reshape(n,2,-1),messages=messages,
                    message_logits=classifier(messages))

    def save_state(state,name):
        torch.save(dict(example_ids=[r.example_id for r in rows],agent_order=[0,1],**state),
                   args.out/(name+'.pt'))

    @torch.inference_mode()
    def project(messages):
        return original._project_all(channel,messages.reshape(-1,channel.message_slots,channel.message_dim),
                                     batch_size=diagnostic_batch,device=device)

    with torch.inference_mode():
        summary['decoder_parity']=author.verify_decoder_parity(*common_args,
            output_position_offset=int(mc['output_position_offset']),device=device)
        setup_seconds=time.monotonic()-started
        native=observe()
        save_state(native,'native')
        runs={}
        for route in ('cross','self'):
            run=[native]
            for depth in range(1,7):
                incoming=original._incoming_messages(run[-1]['messages'],
                    condition='matched' if route=='cross' else 'self')
                prefixes=project(incoming).reshape(n,2,channel.message_slots,channel.model_dim)
                run.append(observe(prefixes))
                save_state(run[-1],f'{route}_{depth}')
                print(json.dumps(dict(route=route,round=depth,examples=n,
                    elapsed_seconds=time.monotonic()-started)),flush=True)
            runs[route]=run

        # G0(m0),G1(m0),G0(m1),G1(m1) already exist in cross/self round1.
        contrasts=[]
        for source in (0,1):
            a=runs['self' if source==0 else 'cross'][1]['messages'][:,0]
            b=runs['cross' if source==0 else 'self'][1]['messages'][:,1]
            ca=runs['self' if source==0 else 'cross'][1]['message_logits'][:,0]
            cb=runs['cross' if source==0 else 'self'][1]['message_logits'][:,1]
            if source==0:
                la=runs['self'][2]['label_logits'][:,0]
                lb=runs['cross'][2]['label_logits'][:,0]
            else:
                # Same final receiver/context for both re-encoded messages.
                _,_,la=forward(project(a),receiver_only=True)
                _,_,lb=forward(project(b),receiver_only=True)
            x,y=a.float().flatten(1),b.float().flatten(1)
            relative_rms=(x-y).square().mean(-1).sqrt()/((x.square()+y.square()).mean(-1)/2).sqrt().clamp_min(1e-12)
            cosine=(x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-12)
            contrasts.append(dict(source=source,relative_message_rms=relative_rms.tolist(),
                message_cosine=cosine.tolist(),message_classifier_tv=((ca.softmax(-1)-cb.softmax(-1)).abs().sum(-1)/2).tolist(),
                incoming_message_label=native['message_logits'][:,source].argmax(-1).tolist(),
                outgoing_message_labels=torch.stack([ca.argmax(-1),cb.argmax(-1)],dim=1).tolist(),
                common_receiver_logits=torch.stack([la,lb],dim=1).tolist(),
                common_receiver_tv=((la.softmax(-1)-lb.softmax(-1)).abs().sum(-1)/2).tolist()))

        # Separate author calls are needed only on fixed validity cohort.
        checks=[]
        if args.stage=='validity':
            options=dict(derangement_order=author.qasc_derangement_order(rows,salt=config['evaluation']['derangement_salt']),
                batch_examples=int(config['evaluation']['qasc_batch_examples']),
                diagnostic_batch_size=diagnostic_batch,output_position_offset=int(mc['output_position_offset']),
                random_seed=int(config['evaluation']['random_message_seed']),
                no_message_receiver_batch_examples=8,device=device)
            for condition,depth in [('matched',1),('matched',4),('self',1),('self',4),('final_only',4),('no_message',4)]:
                expected,metadata=author.evaluate_condition(channel,*common_args,
                    condition=condition,exchange_rounds=depth,**options)
                target=native if condition=='no_message' else runs[
                    'self' if condition=='self' else 'cross'][1 if condition=='final_only' else depth]
                predicted=target['predictions'][:,0]
                check=dict(condition=condition,rounds=depth,
                           predictions_equal=bool(torch.equal(expected,predicted)))
                if condition in ('matched','self'):
                    hashes=[sha256_tensor(v['messages']) for v in runs['cross' if condition=='matched' else 'self'][:depth]]
                    check['message_hashes_equal']=hashes==metadata['round_message_sha256']
                checks.append(check)
            summary['independent_author_checks']=checks

        references={r['example_id']:r for r in map(json.loads,(base/'evidence/qasc_predictions.jsonl').read_text().splitlines())}
        labels=config['surface']['qasc_labels']
        ref_conditions={'final_only':runs['cross'][1],'matched':runs['cross'][4],
                        'self':runs['self'][4],'no_message':native}
        summary['reference_agreement']={}
        for name,state in ref_conditions.items():
            mismatches=[row.example_id for i,row in enumerate(rows)
                if labels[int(state['predictions'][i,0])]!=references[row.example_id]['conditions'][name]['label']]
            summary['reference_agreement'][name]=dict(equal=n-len(mismatches),total=n,mismatch_ids=mismatches)
        with (args.out/'per_item.jsonl').open('x') as handle:
            for i,row in enumerate(rows):
                def compact(state):
                    return {key:state[key][i].tolist() for key in ('predictions','label_logits','message_logits')}
                record=dict(example_id=row.example_id,gold=labels.index(row.answer_label),
                    native=compact(native),cross=[compact(s) for s in runs['cross'][1:]],
                    self=[compact(s) for s in runs['self'][1:]],
                    common_input=[{key:(value if key=='source' else value[i]) for key,value in c.items()} for c in contrasts])
                handle.write(json.dumps(record)+'\n')
        summary.update(completed=True,examples=n,setup_seconds=setup_seconds,
            elapsed_seconds=time.monotonic()-started,forward_seconds=forward_seconds,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            numerical_validity_passed=all(c['predictions_equal'] and c.get('message_hashes_equal',True) for c in checks),
            public_reference_all_equal=all(r['equal']==n for r in summary['reference_agreement'].values()))
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if not summary['numerical_validity_passed'] or not summary['public_reference_all_equal']:
            raise SystemExit('Saved mismatches; resolve numerical/reproduction differences before mechanism claims')
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
