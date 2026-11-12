"""Gates for the evidenced terminal-readout correction; never bless old v2."""
if __package__:
    from .dream_trace_gate_v2 import FINGERPRINT_KEYS, implementation_checks_pass
else:
    from dream_trace_gate_v2 import FINGERPRINT_KEYS, implementation_checks_pass

POLICY = 'within_environment_v3_corrected_readout'


def validate_diagnostic(legacy, diagnostic):
    assert legacy['stage'] == 'full' and legacy['completed'] and legacy['examples'] == 1000
    assert legacy['execution_policy'] == 'within_environment_v2'
    assert legacy['numerical_validity_passed'] is False
    assert len(legacy['independent_author_checks']) == 6
    messages = [c for c in legacy['independent_author_checks'] if c['condition'] in ('matched', 'self')]
    assert len(messages) == 4 and all(c['message_hashes_equal'] for c in messages)
    assert diagnostic['completed'] and diagnostic['examples'] == 1000
    assert diagnostic['stage'] == 'readout_diagnostic' and diagnostic['is_validity_receipt'] is False
    assert diagnostic['numerical_flags'] == legacy['numerical_flags']
    assert diagnostic['joint_replay_hidden_exact']
    assert diagnostic['fresh_author_native_message_hash_equal']
    assert diagnostic['final_only_projection_equals_joint_receiver_slice']
    for key in ('cross1_joint_replay_vs_saved', 'cross1_cached_receiver_vs_fresh_author',
                'native_receiver_full_vs_decoder'):
        check = diagnostic['comparisons'][key]
        assert check['equal'] == check['total'] == 1000
        assert not check['mismatch_ids']
        if key != 'cross1_cached_receiver_vs_fresh_author':
            assert check['max_abs_label_logit_difference'] == 0
    expected_counts = {'cross1_receiver_vs_saved_joint': 998,
                       'cross4_receiver_vs_saved_joint': 996,
                       'self1_receiver_vs_saved_joint': 999,
                       'self4_receiver_vs_saved_joint': 1000,
                       'final_only_receiver_vs_saved_joint': 998,
                       'native_receiver_full_vs_saved_joint': 988}
    for key, count in expected_counts.items():
        assert diagnostic['comparisons'][key]['equal'] == count


def project_corrected_full_seconds(legacy, diagnostic):
    validate_diagnostic(legacy, diagnostic)
    assert len(legacy['forward_seconds']) == 15
    author_and_other = legacy['elapsed_seconds'] - legacy['setup_seconds'] - sum(legacy['forward_seconds'])
    assert author_and_other > 0
    receiver = max(legacy['forward_seconds'][-2:]) * 1.2
    seconds = 1.5 * (legacy['setup_seconds'] + author_and_other + 30 * receiver) + 180
    assert seconds <= 7200
    assert diagnostic['torch_peak_reserved_bytes'] < 22 * 1024**3
    return seconds


def validate_probe_baseline(prior, current):
    assert prior['completed'] and prior['stage'] == 'full' and prior['examples'] == 1000
    assert prior['execution_policy'] == POLICY
    assert prior['numerical_validity_passed'] is True
    assert implementation_checks_pass(prior['independent_author_checks'])
    for key in (*FINGERPRINT_KEYS, 'legacy_full_summary_sha256', 'diagnostic_summary_sha256'):
        assert prior[key] == current[key], key
    assert prior['numerical_flags']['bf16_reduced_precision_reduction'] is True


def project_probe_seconds(prior, legacy):
    assert prior['completed'] and prior['examples']==1000
    assert prior['numerical_validity_passed'] is True
    assert implementation_checks_pass(prior['independent_author_checks'])
    assert len(prior['terminal_seconds'])==26 and len(prior['forward_seconds'])==4
    assert len(legacy['forward_seconds'])==15
    variable=sum(legacy['forward_seconds'][:13])+sum(prior['terminal_seconds'])+sum(prior['forward_seconds'])
    estimate=1.3*(prior['setup_seconds']+192/1000*variable)+120
    assert estimate<=1200
    assert prior['torch_peak_reserved_bytes']<22*1024**3
    return estimate
