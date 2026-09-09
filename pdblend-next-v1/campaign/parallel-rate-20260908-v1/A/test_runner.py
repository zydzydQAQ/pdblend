import runner

def test_engineering_gate_refuses_503_even_when_measurement_valid():
    assert not runner.engineering_assessment([{'http_status': '503', 'error': 'unavailable'}], {'work_complete': True})['passed']

def test_engineering_gate_refuses_truncated_frequency_fault():
    assert not runner.engineering_assessment([{'error': 'Frequency recovery failed'}], {'work_complete': True})['passed']

def test_regular_capacity_miss_is_kept_but_first_gate_requires_complete_work():
    rows = [{'error': 'request deadline exceeded', 'http_status': ''}]
    summary = {'work_complete': False, 'slo_attainment': 0.1}
    assert runner.engineering_assessment(rows, summary)['passed']
    assert not runner.engineering_assessment(rows, summary, True)['passed']

def test_low_slo_is_not_an_engineering_error():
    assert runner.engineering_assessment([{'error': '', 'http_status': '200'}], {'work_complete': True, 'slo_attainment': 0.1}, True)['passed']

def test_fresh_A_declaration_preserves_exact_original_work():
    d = runner.p.read(runner.p.ROOT / 'work-declaration.json')
    assert len(d['cells']) == 13
    assert runner.p.sha(runner.p.ROOT / 'work-declaration.json') == runner.DECLARATION_SHA
    for c in d['cells']:
        assert c['model'] == '14b'
        original = runner.p.checked(c['original_checkpoint'])['row']
        assert original == c['source_row']
        assert runner.p.sha(c['trace']['path']) == c['trace']['sha256']
