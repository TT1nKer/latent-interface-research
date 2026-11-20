"""Build the preregistered LEAPS cohort with native validation and cross-bank deduplication."""
import argparse
from collections import Counter
import json
from pathlib import Path

from leaps_karel_source import run as validate_native, sha_bytes, sha_file, write


def code_for(row):
    digest = sha_bytes(' '.join(row['program_tokens']).encode())
    if digest != row['code']:
        raise ValueError('Previously normalized code hash does not match tokens')
    return digest


def prepare(protocol_path, development, prior_leaps, folder, report):
    protocol = json.loads(protocol_path.read_text())
    prior = protocol['prior_sources']
    required = {
        prior_leaps: prior['prior_leaps_normalized_sha256'],
        development: prior['development_program_pairs_sha256'],
        Path(__file__).with_name('leaps_karel_source.py'): prior['prior_acquirer_sha256'],
        Path(__file__).with_name('leaps_karel_source_format.py'): prior['prior_format_sha256'],
    }
    for path, fingerprint in required.items():
        if sha_file(path) != fingerprint:
            raise ValueError(f'Input fingerprint mismatch: {path}')
    prior_ids, prior_codes = set(), set()
    for line in prior_leaps.read_text().splitlines():
        row = json.loads(line)
        prior_ids.add(row['guid'])
        prior_codes.add(code_for(row))
    development_codes = {json.loads(line)['code'] for line in development.read_text().splitlines()}
    if (len(prior_codes) != prior['prior_leaps_fixed_count'] or
            len(prior_ids) != prior['prior_leaps_fixed_count'] or
            len(development_codes) != prior['development_unique']):
        raise ValueError('Reference program count mismatch')
    folder.mkdir(mode=0o700,parents=True,exist_ok=True)
    (folder/'.gitignore').write_text('*\n')
    native_report = report.with_name('native_summary.json')
    native_dir = folder/'native'
    if not native_report.exists():
        validate_native(protocol_path, development, native_dir, native_report)
    native = json.loads(native_report.read_text())
    if not native.get('completed') or not native.get('native_validity_passed'):
        raise ValueError('Prior native verification was incomplete; no reselection permitted')
    if (sha_file(native_dir/'cohort.jsonl') != native['artifact_sha256'] or
            sha_file(protocol_path) != native['protocol_sha256']):
        raise ValueError('Native validation source or protocol fingerprint mismatch')
    expected = [i for s in protocol['sampling']['block_starts']
                for i in range(s,s+protocol['sampling']['block_length'])]
    counts = Counter()
    emitted = []
    seen = set()
    outfile = folder/'cohort.jsonl'
    partial = folder/'cohort.jsonl.partial'
    audit = folder/'selection_audit.jsonl'
    with (native_dir/'cohort.jsonl').open() as source, partial.open('x') as output, audit.open('x') as log:
        for index,line in zip(expected,source,strict=True):
            row = json.loads(line)
            if row['source_index'] != index or row['guid'] in prior_ids:
                raise ValueError('Selected source order or ID disjointness mismatch')
            digest = code_for(row)
            reasons = []
            if digest in development_codes:
                reasons.append('development_code')
            if digest in prior_codes:
                reasons.append('prior_leaps_code')
            if digest in seen:
                reasons.append('within_cohort_code')
            seen.add(digest)
            demos = row['examples']
            if list(range(6)) != [case['example_index'] for case in demos]:
                raise ValueError('Native world order mismatch')
            zeros = sum(not case['actions'] for case in demos)
            no_move = sum('move' not in case['actions'] for case in demos)
            is_eligible = not reasons
            for group in ['all_selected', *(['eligible'] if is_eligible else [])]:
                counts.update({group+'_programs':1,group+'_worlds':len(demos),
                               group+'_zero_action_worlds':zeros,group+'_no_move_worlds':no_move,
                               group+'_programs_all_worlds_zero_action':int(zeros == 6),
                               group+'_programs_any_world_zero_action':int(zeros > 0),
                               group+'_programs_all_worlds_no_move':int(no_move == 6),
                               group+'_programs_any_world_no_move':int(no_move > 0)})
            for reason in reasons:
                counts['excluded_'+reason] += 1
            log.write(json.dumps([row['guid'],digest,reasons])+'\n')
            if is_eligible:
                if not row['eligible']:
                    raise ValueError('Earlier native eligibility disagrees')
                emitted.append(digest)
                output.write(line)
    if not (counts['all_selected_programs'] == native['selected_programs'] == len(expected) and
            counts['all_selected_worlds'] == native['validated_stop'] == len(expected)*6 and
            counts['all_selected_no_move_worlds'] == native['no_move_worlds'] and
            counts['all_selected_zero_action_worlds'] == native['zero_action_worlds'] and
            len(set(emitted)) == len(emitted) and not (set(emitted) & (prior_codes|development_codes))):
        raise ValueError('Final source accounting check failed')
    partial.replace(outfile)
    final = dict(native)
    final.update({k:counts[k] for k in ['excluded_prior_leaps_code','excluded_development_code',
                                       'excluded_within_cohort_code']})
    final.update(dict(counts))
    final.update({'prior_leaps_artifact_sha256':sha_file(prior_leaps),'native_gate_sha256':sha_file(native_report),
                  'wrapper_sha256':sha_file(Path(__file__)),'artifact_bytes':outfile.stat().st_size,
                  'artifact_sha256':sha_file(outfile),'eligible_programs':len(emitted),
                  'eligible_ordered_codes_sha256':sha_bytes(('\n'.join(emitted)+'\n').encode()),
                  'source_ids_overlap_prior_leaps':0,'eligible_exact_overlap_prior_leaps':0,
                  'eligible_exact_overlap_development':0,'audit_sha256':sha_file(audit)})
    write(report, final)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('protocol','development','prior_leaps','folder','report'):
        cli.add_argument(arg,type=Path)
    args = cli.parse_args()
    prepare(args.protocol,args.development,args.prior_leaps,args.folder,args.report)
