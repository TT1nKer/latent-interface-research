"""Fail-closed gates for explicitly amended, within-environment experiments."""

EXPECTED_CHECKS = {('matched', 1), ('matched', 4), ('self', 1),
                   ('self', 4), ('final_only', 4), ('no_message', 4)}
FINGERPRINT_KEYS = ('code_sha256', 'gate_sha256', 'amendment_sha256',
                    'protocol_sha256', 'revision', 'torch', 'cuda', 'gpu',
                    'data_sha256', 'checkpoint_sha256', 'config_sha256', 'reference_sha256')


def implementation_checks_pass(checks):
    if len(checks) != len(EXPECTED_CHECKS):
        return False
    if {(c['condition'], c['rounds']) for c in checks} != EXPECTED_CHECKS:
        return False
    return all(c.get('predictions_equal') is True and
               (c['condition'] not in ('matched', 'self') or
                c.get('message_hashes_equal') is True) for c in checks)


def validate_prior(prior, current, expected_stage):
    assert prior['completed'] and prior['stage'] == expected_stage
    assert prior['execution_policy'] == 'within_environment_v2'
    assert prior['examples'] == (16 if expected_stage == 'validity' else 1000)
    assert prior['numerical_validity_passed'] is True
    assert implementation_checks_pass(prior['independent_author_checks'])
    for key in FINGERPRINT_KEYS:
        assert prior[key] == current[key], f'Prior receipt fingerprint changed: {key}'
    # Exact public-reference agreement remains reported, but is not the
    # implementation gate in this explicitly amended experiment.
    assert isinstance(prior['public_reference_all_equal'], bool)


def project_full_seconds(prior):
    assert prior['examples'] == 16
    setup = prior['setup_seconds']
    measured = prior['elapsed_seconds']
    assert 0 <= setup < measured
    # Includes all six independent author checks on all1000, not just tracing.
    return 1.5 * (setup + (measured - setup) * 1000 / 16) + 180


def full_resource_gate(prior, amendment):
    estimate = project_full_seconds(prior)
    assert estimate <= amendment['budgets']['full_wall_seconds'], (
        f'Conservative empirical projection {estimate:.1f}s exceeds full budget')
    assert prior['torch_peak_reserved_bytes'] < 22 * 1024**3
    return estimate
