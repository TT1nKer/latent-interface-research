"""Verify the repaired binary package and text boundaries; never imports models/torch."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from packaging.requirements import Requirement
from tokenizers import Tokenizer
from late_binding_tasks import make_case, validate


def main():
    # Metadata-only dependency closure check; avoids loading torch or models on
    # login nodes while catching missing/incompatible transitive requirements.
    pending = ['transformers', 'tokenizers', 'huggingface-hub']
    dependencies = {}
    while pending:
        name = pending.pop()
        if name in dependencies:
            continue
        version = importlib.metadata.version(name)
        dependencies[name] = version
        for raw in importlib.metadata.requires(name) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            installed = importlib.metadata.version(requirement.name)
            assert requirement.specifier.contains(installed), (name, raw, installed)
            pending.append(requirement.name)
    versions = {n:importlib.metadata.version(n) for n in ('transformers','tokenizers','huggingface-hub')}
    req = next(Requirement(s) for s in importlib.metadata.requires('transformers')
               if Requirement(s).name == 'tokenizers')
    assert req.specifier.contains(versions['tokenizers']), (str(req), versions)
    acquisition = json.loads(Path('results/c2c_acquisition_v1.json').read_text())
    assert acquisition['passed']
    paths = {m['repo']:Path(m['snapshot']) for m in acquisition['models']}
    tokenizers = [Tokenizer.from_file(str(paths[name]/'tokenizer.json'))
        for name in ('Qwen/Qwen3-0.6B','Qwen/Qwen2.5-0.5B-Instruct')]
    encode = lambda t,s:t.encode(s, add_special_tokens=False).ids
    label_ids = [encode(tokenizers[0],' '+letter) for letter in 'ABCD']
    assert all(len(x)==1 for x in label_ids)
    response = encode(tokenizers[0],'The correct answer is')
    rows = json.loads(Path('late_binding_arc_dev32_v1.json').read_text())['rows']
    validate(rows)
    checked = 0
    for row in rows:
        for shift in range(4):
            c = make_case(row,shift)
            prompt = c['prefix']+c['suffix']
            ids = encode(tokenizers[0],prompt)
            assert ids == encode(tokenizers[1],prompt)
            prefix_ids = encode(tokenizers[0],c['prefix'])
            assert ids[:len(prefix_ids)] == prefix_ids
            assert ids[-len(response):] == response
            for i,letter in enumerate('ABCD'):
                assert encode(tokenizers[0],prompt+' '+letter) == ids+label_ids[i]
            checked += 1
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    report = dict(passed=True,versions=versions,dependency_closure=dependencies,
        transformers_tokenizer_requirement=str(req),
        checked_prompts=checked,label_ids=label_ids,scope='Binary tokenizer load and text boundary checks only; no model execution.',
        preflight_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path('results/c2c_tokenizer_preflight_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
