"""Correct terminal readouts; reuse full-v2 recurrence or recompute the probe.

Old trajectories, failed receipts and frozen scientific statistics are preserved.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from dream_readout_gate_v3 import (POLICY, validate_diagnostic,
    project_corrected_full_seconds, validate_probe_baseline, project_probe_seconds,
    implementation_checks_pass)


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
    parser.add_argument('--stage',choices=['full','sensitivity'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-receipt',type=Path)
    args=parser.parse_args()
    assert torch.cuda.is_available()
    assert 'L4' in torch.cuda.get_device_name()
    args.out.mkdir(parents=True,exist_ok=False)
    base=Path('vendor/hidden_messages_f85b6d8')
    revision=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()
    assert revision=='f85b6d8563a2236117407e0df9c6cd60ea406dac'
    subprocess.run(['git','-C',str(base),'diff','--quiet','HEAD'],check=True)
    protocol=json.loads(Path('dream_trace_protocol_v1.json').read_text())
    amendment=json.loads(Path('dream_readout_amendment_v3.json').read_text())
    assert amendment['policy']==POLICY
    assert sha('dream_trace_protocol_v1.json')==amendment['base_protocol_sha256']
    assert sha('dream_recurrent_trace_v2.py')==amendment['base_driver_v2_sha256']
    assert sha('dream_trace_amendment_v2.json')==amendment['base_amendment_v2_sha256']
    legacy_dir=Path('results/dream_trace_full_v2_12448809')
    diagnostic_dir=Path('results/dream_readout_diagnostic_v1_12454532')
    assert sha(legacy_dir/'summary.json')==amendment['legacy_full_summary_sha256']
    assert sha(diagnostic_dir/'summary.json')==amendment['diagnostic_summary_sha256']
    assert sha('dream_readout_diagnostic_v1.py')==amendment['diagnostic_code_sha256']
    legacy=json.loads((legacy_dir/'summary.json').read_text())
    assert sha('dream_trace_gate_v2.py')==legacy['gate_sha256']
    diagnostic=json.loads((diagnostic_dir/'summary.json').read_text())
    validate_diagnostic(legacy,diagnostic)
    projected=project_corrected_full_seconds(legacy,diagnostic)
    prior=None
    receipt=args.baseline_receipt
    if args.stage=='sensitivity':
        assert receipt, 'Sensitivity requires a matching corrected full receipt'
        prior=json.loads(receipt.read_text())
    def numerical_flags():
        return dict(bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    original_flags=numerical_flags()
    assert original_flags==legacy['numerical_flags']
    if args.stage=='sensitivity':
        assert prior['numerical_flags']==original_flags
        assert original_flags['bf16_reduced_precision_reduction'] is True, 'Probe would not change this setting'
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    config=author.load_config(base/'configs/qasc_original_evaluation.yaml')
    data=base/'data/qasc/validation.jsonl'
    ckpt=base/'checkpoints/qasc_dense_latent_prefix_channel.pt'
    assert sha(data)==config['qasc']['development_sha256']
    assert sha(ckpt)==config['post_hoc']['source_channel_sha256']
    started=time.monotonic()
    summary=dict(job_id=os.environ['SLURM_JOB_ID'],stage=args.stage,revision=revision,
        code_sha256=sha(__file__),protocol_sha256=sha('dream_trace_protocol_v1.json'),
        data_sha256=sha(data),checkpoint_sha256=sha(ckpt),completed=False,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        execution_policy=POLICY,amendment_sha256=sha('dream_readout_amendment_v3.json'),
        gate_sha256=sha('dream_readout_gate_v3.py'),config_sha256=sha(base/'configs/qasc_original_evaluation.yaml'),
        reference_sha256=sha(base/'evidence/qasc_predictions.jsonl'),
        numerical_flags=numerical_flags(),original_numerical_flags=original_flags,
        reference_replication_required_for_execution=False,
        legacy_full_summary_sha256=sha(legacy_dir/'summary.json'),
        diagnostic_summary_sha256=sha(diagnostic_dir/'summary.json'),
        terminal_readout='Separate receiver-only author readout for each agent; native full-model path',
        recurrence_reused=args.stage=='full',state_file_sha256={},
        projected_full_seconds_with_margin=projected)
    for key in ('revision','torch','cuda','gpu','data_sha256','checkpoint_sha256','config_sha256','reference_sha256','protocol_sha256'):
        assert summary[key]==legacy[key],key
    if prior:
        validate_probe_baseline(prior,summary)
        summary['projected_probe_seconds_with_margin']=project_probe_seconds(prior,legacy)

    if prior:
        summary['prior_validity_receipt_sha256']=sha(receipt)
        summary['numerical_validity_scope']='Numerical-setting probe; corrected baseline separately verified on all1000'
    else:
        summary['numerical_validity_scope']='All1000 separate author calls against corrected readouts and cached recurrence messages'
    (args.out/'started.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=author.load_qasc_rows(data,expected_rows=1000,expected_split='validation')
    if args.stage=='sensitivity':
        rows=[row for start in amendment['sensitivity']['block_starts']
              for row in rows[start:start+amendment['sensitivity']['block_size']]]
    n=len(rows)
    summary['example_ids_sha256']=hashlib.sha256('\n'.join(r.example_id for r in rows).encode()).hexdigest()
    mc=config['model']
    snapshot=Path('hf_cache/hub')/('models--'+mc['id'].replace('/','--'))/'snapshots'/mc['revision']
    manifest,_=author.model_weight_manifest(snapshot)
    assert manifest==mc['weight_manifest_sha256']
    assert manifest==legacy['model_weight_manifest_sha256']
    summary['model_weight_manifest_sha256']=manifest
    if prior:
        assert prior['model_weight_manifest_sha256']==manifest
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
    legacy_tokens={r['example_id']:r['agent_token_ids'] for r in map(json.loads,(legacy_dir/'token_rows.jsonl').read_text().splitlines())}
    assert all([list(t) for t in tokens]==legacy_tokens[row.example_id]
               for row,tokens in zip(rows,cohort.token_rows,strict=True))
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

    terminal_seconds=[]

    @torch.inference_mode()
    def terminal_forward(prefixes,agent):
        read_cohort=cohort if agent==0 else replace(cohort,
            token_rows=tuple((pair[1],pair[0]) for pair in cohort.token_rows))
        captured=[]
        cursor=0
        def capture(_module,_inputs,logits):
            nonlocal cursor
            if logits.ndim==3:
                lengths=torch.tensor([len(pair[0]) for pair in
                    read_cohort.token_rows[cursor:cursor+len(logits)]],device=device)
                positions=lengths-1+int(mc['output_position_offset'])
                logits=logits[torch.arange(len(logits),device=device),positions]
            assert logits.ndim==2
            cursor+=len(logits)
            captured.append(logits.index_select(-1,label_ids).float().cpu())
        hook=head.register_forward_hook(capture)
        torch.cuda.synchronize()
        t=time.monotonic()
        try:
            _,predictions=original._forward_cohort(model,decoder,embedding,head,tokenizer,read_cohort,
                batch_examples=int(config['evaluation']['qasc_batch_examples']),
                output_position_offset=int(mc['output_position_offset']),prefixes=prefixes,
                receiver_only=True,reference_full_model=prefixes is None,device=device)
        finally:
            hook.remove()
        torch.cuda.synchronize()
        terminal_seconds.append(time.monotonic()-t)
        logits=torch.cat(captured)
        assert cursor==n and torch.equal(logits.argmax(-1),predictions)
        assert torch.isfinite(logits).all()
        return predictions,logits

    def cached_state(name):
        path=legacy_dir/(name+'.pt')
        digest=sha(path)
        if name in diagnostic['state_file_sha256']:
            assert digest==diagnostic['state_file_sha256'][name]
        summary['state_file_sha256'][name]=digest
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['example_ids']==[r.example_id for r in rows] and state['agent_order']==[0,1]
        assert state['messages'].shape[:2]==(n,2)
        assert torch.equal(state['label_logits'].argmax(-1),state['predictions'])
        return state

    with torch.inference_mode():
        summary['decoder_parity']=author.verify_decoder_parity(*common_args,
            output_position_offset=int(mc['output_position_offset']),device=device)
        setup_seconds=time.monotonic()-started
        if args.stage=='full':
            native=cached_state('native')
            runs={route:[native]+[cached_state(f'{route}_{depth}') for depth in range(1,7)]
                  for route in ('cross','self')}
        else:
            # Recompute all recurrence states under the changed numerical flag.
            # Reusing full-baseline messages would not test that intervention.
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
                runs[route]=run
        summary['recurrence_seconds']=time.monotonic()-started-setup_seconds

        summary['terminal_label_changes_from_joint']=[]
        for route,depths in [('native',[0]),('cross',range(1,7)),('self',range(1,7))]:
            for depth in depths:
                state=native if route=='native' else runs[route][depth]
                joint_predictions=state['predictions'].clone()
                if route=='native':
                    prefixes=None
                else:
                    incoming=original._incoming_messages(runs[route][depth-1]['messages'],
                        condition='matched' if route=='cross' else 'self')
                    prefixes=project(incoming).reshape(n,2,channel.message_slots,channel.model_dim)
                readings=[terminal_forward(None if prefixes is None else prefixes[:,agent],agent)
                          for agent in (0,1)]
                state['predictions']=torch.stack([r[0] for r in readings],dim=1)
                state['label_logits']=torch.stack([r[1] for r in readings],dim=1)
                summary['terminal_label_changes_from_joint'].append(dict(route=route,depth=depth,
                    changed_by_agent=[int((joint_predictions[:,agent]!=state['predictions'][:,agent]).sum())
                                      for agent in (0,1)]))
                print(json.dumps(dict(terminal_route=route,round=depth,examples=n,
                    elapsed_seconds=time.monotonic()-started)),flush=True)

        # G0(m0),G1(m0),G0(m1),G1(m1) already exist in cross/self round1.
        contrasts=[]
        for source in (0,1):
            a=runs['self' if source==0 else 'cross'][1]['messages'][:,0]
            b=runs['cross' if source==0 else 'self'][1]['messages'][:,1]
            ca=runs['self' if source==0 else 'cross'][1]['message_logits'][:,0]
            cb=runs['cross' if source==0 else 'self'][1]['message_logits'][:,1]
            # Both comparisons use the same designated receiver/readout path.
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

        # Independent full1000 author calls remain mandatory after correction.
        checks=[]
        author_predictions={}
        if args.stage=='full':
            options=dict(derangement_order=author.qasc_derangement_order(rows,salt=config['evaluation']['derangement_salt']),
                batch_examples=int(config['evaluation']['qasc_batch_examples']),
                diagnostic_batch_size=diagnostic_batch,output_position_offset=int(mc['output_position_offset']),
                random_seed=int(config['evaluation']['random_message_seed']),
                no_message_receiver_batch_examples=8,device=device)
            for condition,depth in [('matched',1),('matched',4),('self',1),('self',4),('final_only',4),('no_message',4)]:
                expected,metadata=author.evaluate_condition(channel,*common_args,
                    condition=condition,exchange_rounds=depth,**options)
                author_predictions[f'{condition}{depth}']=expected.tolist()
                target=native if condition=='no_message' else runs[
                    'self' if condition=='self' else 'cross'][1 if condition=='final_only' else depth]
                predicted=target['predictions'][:,0]
                check=dict(condition=condition,rounds=depth,
                           predictions_equal=bool(torch.equal(expected,predicted)),
                           mismatch_ids=[rows[i].example_id for i in (expected!=predicted).nonzero().flatten().tolist()])
                if condition in ('matched','self'):
                    hashes=[sha256_tensor(v['messages']) for v in runs['cross' if condition=='matched' else 'self'][:depth]]
                    check['message_hashes_equal']=hashes==metadata['round_message_sha256']
                checks.append(check)
                print(json.dumps(dict(author_check=check)),flush=True)
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
                    independent_author_predictions={key:value[i] for key,value in author_predictions.items()},
                    common_input=[{key:(value if key=='source' else value[i]) for key,value in c.items()} for c in contrasts])
                handle.write(json.dumps(record)+'\n')
        summary.update(completed=True,examples=n,setup_seconds=setup_seconds,
            elapsed_seconds=time.monotonic()-started,forward_seconds=forward_seconds,
            terminal_seconds=terminal_seconds,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            numerical_validity_passed=implementation_checks_pass(checks) if args.stage!='sensitivity' else None,
            public_reference_all_equal=all(r['equal']==n for r in summary['reference_agreement'].values()))
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if args.stage!='sensitivity' and not summary['numerical_validity_passed']:
            raise SystemExit('Saved results; same-environment author implementation comparison failed')
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
