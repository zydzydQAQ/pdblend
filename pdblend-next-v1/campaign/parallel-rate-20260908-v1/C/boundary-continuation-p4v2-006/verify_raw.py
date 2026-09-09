import csv, hashlib, json, math, time
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def verify(cp):
    c = read(cp)
    row = c['row']
    receipt = read(c['receipt'])
    s = receipt['summary']
    refs = dict(c['artifacts'])
    refs[c['receipt']] = c['receipt_sha256']
    refs[c['binding']] = c['binding_sha256']
    refs[row['trace_path']] = row['trace_sha256']
    for (p, h) in refs.items():
        assert sha(p) == h, 'artifact hash mismatch ' + p
    cell = next((Path(p).parent for p in c['artifacts'] if '/cells/' in p and p.endswith('/power.csv')))
    power = list(csv.DictReader((cell / 'power.csv').open()))
    points = [(float(v['t_s']), [float(v[f'gpu{g}_w']) for g in range(8)]) for v in power]
    (start, end) = (s['measurement_start_s'], s['measurement_end_s'])
    assert points[0][0] <= start < end <= points[-1][0]
    energy = [0.0] * 8
    for ((ta, pa), (tb, pb)) in zip(points, points[1:]):
        assert tb > ta and all((math.isfinite(x) and x>=0 for x in pa + pb))
        (a, z) = (max(start, ta), min(end, tb))
        if z <= a:
            continue
        for g in range(8):
            wa = pa[g] + (pb[g] - pa[g]) * (a - ta) / (tb - ta)
            wz = pa[g] + (pb[g] - pa[g]) * (z - ta) / (tb - ta)
            energy[g] += (z - a) * (wa + wz) / 2
    requests = read(row['trace_path'])['requests']
    bench = list(csv.DictReader((cell / 'bench.csv').open()))
    assert len(bench) == len(requests) == row['n_requests'] and len({v['idx'] for v in bench}) == len(bench)
    completed = good = failed = timeouts = generated = 0
    for v in bench:
        req = requests[int(v['idx'])]
        ok = v['success'] == '1' and v['token_ids_verified'] in ('1','True','true') and v['token_count_source']=='server_usage' and int(v['input_tokens'] or 0) == req['prompt_len'] and (int(v['generated_tokens'] or 0) == req['output_len']) and (v['request_timeout'] == 'False')
        completed += ok
        failed += v['success'] != '1'
        timeouts += v['request_timeout'] == 'True'
        generated += int(v['generated_tokens'] or 0)
        good += bool(ok and 0 <= float(v['ttft_s']) < row['slo_ttft_s'] and (0 <= float(v['tpot_s']) < row['slo_tpot_s']))
    assert completed == s['completed_work_requests'] and good == s['good_requests']
    assert failed == s['failed_requests'] and timeouts == s['request_timeouts'] and (generated == s['generated_tokens'])
    assert abs(sum(energy) - s['energy_j']) < 0.0001
    assert receipt['child_stopped'] and (not receipt['outer_cleanup_errors']) and receipt['clock_restore_complete']
    return dict(captured_s=time.time(), checkpoint=str(cp), checkpoint_sha256=sha(cp), cell_id=row['cell_id'], system=row['system'], dataset=row['dataset'], rate=row['rate_rps_decimal'], repeat=row['repeat'], raw_verified=True, files_verified=len(refs), energy_per_gpu_j=energy, energy_j=sum(energy), receipt_energy_delta_j=sum(energy) - s['energy_j'], completed_requests=completed, expected_requests=len(requests), failed_requests=failed, request_timeouts=timeouts, generated_tokens=generated, good_requests=good, slo=good / len(requests), work_complete=completed == len(requests), cleanup_verified=True, gpu_actions=False)
