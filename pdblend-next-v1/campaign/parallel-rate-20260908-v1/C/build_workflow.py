"""Create a new C-only campaign; never edit frozen predecessors."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = REPO / 'campaign/main-slo-improvement-v7'

def write(path, value):
    with path.open('x') as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')

def main():
    old = json.loads((OLD / 'work-declaration.json').read_text())
    cells = [copy.deepcopy(c) for c in old['cells']
             if c['model'] == '7b' and c['arm'] == 'fixed2']
    priority = {('alpaca', 9.): 0, ('alpaca', 12.): 1,
                ('sharegpt', 2.): 2, ('sharegpt', 3.): 3,
                ('longbench', 1.5): 4, ('longbench', 3.): 5}
    cells.sort(key=lambda c: (c['repeat'], priority[(c['dataset'], c['original_point']['rate_rps'])]))
    for c in cells:
        c['cell_id'] = c['cell_id'].replace('slo-improve-v7-', 'parallel-rate-p1-')
    assert len(cells) == 12
    declaration = dict(old, schema='parallel-rate-C-work-p1', implementation_series='parallel-rate-p1',
                       deadline_s=1788872770.0400891, cells=cells, fixed2_screen_count=12,
                       dynamic_full_grid_count=0, first_point_is_engineering_gate=True,
                       preserved_original_trace_and_work=True)
    write(HERE / 'work-declaration.json', declaration)
    digest = hashlib.sha256((HERE / 'work-declaration.json').read_bytes()).hexdigest()
    protocol = (OLD / 'protocol.py').read_text().replace('REPO = ROOT.parents[1]', 'REPO = ROOT.parents[2]')
    write(HERE / 'protocol.py', protocol)
    prepare = (OLD / 'prepare_release.py').read_text().replace("'measured_frequency_write_guard_v1': True}",
        "'measured_frequency_write_guard_v1': True, 'observed_first_admission_frequency_v1': True}")
    write(HERE / 'prepare_release.py', prepare)
    runner = (OLD / 'runner.py').read_text().replace("DECLARATION_SHA = '5a3d0be3b05ca68e7a27eec5464d2c34bf80c21ce11b985b8603262f2ddf6234'", f"DECLARATION_SHA = '{digest}'")
    runner = runner.replace('import copy\n', 'import copy\nimport csv\n')
    anchor = 'def point_binding(base, release, cell, output, out):'
    helper = '''def engineering_gate(summary, bench_path, first):
    with Path(bench_path).open() as stream:
        failures = [dict(request_id=r.get('request_id'), error=r.get('error'), http_status=r.get('http_status'))
                    for r in csv.DictReader(stream)
                    if r.get('http_status') == '503' or 'HTTP 503' in r.get('error', '')]
    return dict(passed=not failures and (not first or summary.get('work_complete') is True),
                http503=failures, first_point=first, work_complete=summary.get('work_complete'),
                subsequent_capacity_failures_are_retained=True)

'''
    runner = runner.replace(anchor, helper + anchor)
    marker = "                    state['completed'].append(cell['cell_id'])"
    addition = '''
                    gate = engineering_gate(receipt['summary'], output / 'cells' / cell['cell_id'] / 'bench.csv', len(state['completed']) == 1)
                    p.write(output / 'engineering-gates' / (cell['cell_id'] + '.json'), gate, exclusive=True)
                    if not gate['passed']:
                        state.update(engineering_gate_failed=True, phase='engineering_gate_failed',
                                     engineering_failure_cell=cell['cell_id'], engineering_failure=gate)
                        break'''
    runner = runner.replace(marker, marker + addition)
    runner = runner.replace("state['complete'] = len(state['completed']) == len(cells)",
                            "state['complete'] = len(state['completed']) == len(cells) and not state.get('engineering_gate_failed')")
    write(HERE / 'runner.py', runner)
    write(HERE / 'workflow-lineage.json', dict(parent=str(OLD), files={str(OLD / f): hashlib.sha256((OLD / f).read_bytes()).hexdigest()
          for f in ['protocol.py', 'runner.py', 'prepare_release.py', 'work-declaration.json']},
          declaration_sha256=digest, changed_scope='C twelve original fixed2 cells, first-point engineering gate and stop on HTTP503'))
    print(json.dumps(dict(declaration_sha256=digest, cells=len(cells))))

if __name__ == '__main__':
    main()
