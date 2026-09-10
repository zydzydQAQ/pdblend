"""CPU-only accounting of audited energy and sampled SM clocks; no causal attribution."""
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(reference):
    path = Path(reference['path'])
    assert ref(path)['sha256'] == reference['sha256'], path
    return path


def main():
    rows, evidence = [], []
    for node in ['A', 'C']:
        source = ROOT / node / 'observations.json'
        evidence.append(ref(source))
        for observation in json.loads(source.read_text()):
            audit_path = verified(observation['audit_reference'])
            a = json.loads(audit_path.read_text())
            summary_path = verified(a['summary'])
            s = json.loads(summary_path.read_text())
            cp_path = verified(a['checkpoint'])
            cp = json.loads(cp_path.read_text())
            clocks_path = Path(next(k for k in cp['artifacts'] if k.endswith('/power/clocks.csv')))
            clock_ref = {'path': str(clocks_path), 'sha256': cp['artifacts'][str(clocks_path)]}
            verified(clock_ref)
            start, end = s['measurement_start_s'], s['measurement_end_s']
            duration = end - start
            assert math.isclose(duration, a['measurement_duration_s'], rel_tol=1e-12)
            assert math.isclose(sum(a['energy_per_gpu_j']), a['energy_j'], rel_tol=1e-12)
            clocks = list(csv.DictReader(clocks_path.open()))
            frequencies = [defaultdict(float) for _ in range(8)]
            covered, intervals = 0., 0
            for left, right in zip(clocks, clocks[1:]):
                t0, t1 = float(left['t_s']), float(right['t_s'])
                assert t1 >= t0
                dt = max(0., min(end, t1)-max(start, t0))
                if not dt:
                    continue
                covered += dt
                intervals += 1
                for gpu in range(8):
                    frequencies[gpu][float(left[f'gpu{gpu}_sm_mhz'])] += dt
            assert math.isclose(covered, duration, abs_tol=1e-6), (a['cell_id'], covered, duration)
            gpu_w = [e/duration for e in a['energy_per_gpu_j']]
            zero_util = [i for i,v in enumerate(a['gpu_util_per_gpu']) if v == 0]
            row = dict(node=node, system=a['system'], rate_rps=a['rate_rps'], repeat=a['repeat'],
                       cell_id=a['cell_id'], slo_attainment=a['slo_attainment'],
                       energy_j=a['energy_j'], duration_s=duration, average_all8_power_w=a['energy_j']/duration,
                       energy_per_good_request_j=a['energy_per_good_request_j'],
                       goodput_measurement_rps=a['goodput_measurement_rps'],
                       gpu_average_power_w=gpu_w, gpu_util=a['gpu_util_per_gpu'],
                       zero_sampled_util_gpu_indices=zero_util,
                       zero_sampled_util_gpu_power_w=sum(gpu_w[i] for i in zero_util),
                       other_gpu_power_w=sum(gpu_w[i] for i in range(8) if i not in zero_util),
                       sm_clock_sample_intervals=intervals,
                       sm_clock_covered_s=covered,
                       sampled_sm_clock_time_fraction=[{str(f):t/covered for f,t in sorted(d.items())} for d in frequencies],
                       source_refs=[observation['audit_reference'],a['summary'],a['checkpoint'],clock_ref])
            rows.append(row)
    pairs = []
    for p in rows:
        if p['system'] != 'pdblend' or p['repeat'] != 1:
            continue
        for b in rows:
            if b['node'] != p['node'] or b['rate_rps'] != p['rate_rps'] or b['repeat'] != 1 or b['system']=='pdblend':
                continue
            er = p['energy_j']/b['energy_j']
            pr = p['average_all8_power_w']/b['average_all8_power_w']
            dr = p['duration_s']/b['duration_s']
            assert math.isclose(er, pr*dr, rel_tol=1e-12)
            pairs.append(dict(node=p['node'],rate_rps=p['rate_rps'],baseline=b['system'],
                              pdb_at_least_90=p['slo_attainment']>=.9,
                              pdb_slo=p['slo_attainment'],baseline_slo=b['slo_attainment'],
                              pdb_power_w=p['average_all8_power_w'],baseline_power_w=b['average_all8_power_w'],
                              pdb_duration_s=p['duration_s'],baseline_duration_s=b['duration_s'],
                              energy_ratio=er,power_ratio=pr,duration_ratio=dr,
                              energy_reduction_pct=100*(1-er),average_power_reduction_pct=100*(1-pr),
                              duration_change_pct=100*(dr-1),
                              identity_residual=er-pr*dr))
    assert len(rows)==62 and len(pairs)==48
    data=dict(schema='slo-rate-power-mechanisms-v1',created_s=time.time(),passed=True,
              scope='All 62 new A/C observations; 48 same-host/rate regular PDB-to-baseline pairs. B reference not reanalysed here.',
              rows=rows,pairs=pairs,sources=evidence,code=ref(Path(__file__)),
              interpretation=['E = average all8 power × measured duration is an accounting identity, not a causal decomposition.',
                              'Per-GPU energy/utilization copied from SHA-verified previously fully replayed audit; no new raw-power reintegration in this analysis.',
                              'SM frequency residency is a left-held sampled estimate clipped to the exact primary window, not exact event-level residence.',
                              'Zero sampled utilization means no activity detected in utilization samples, not proof that every instant was idle.',
                              'Different configured active/resident GPU counts, scheduling, DVFS and timing are jointly varied; separate causal contributions require ablation.'])
    (OUT/'power-mechanisms-evidence.json').write_text(json.dumps(data,indent=2)+'\n')
    for filename, table in [('power-mechanisms.csv',rows),('power-time-pairs.csv',pairs)]:
        with (OUT/filename).open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(table[0]));w.writeheader()
            w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in table)
    print(json.dumps({'passed':True,'observations':len(rows),'pairs':len(pairs),'evidence':ref(OUT/'power-mechanisms-evidence.json')}))


if __name__ == '__main__':
    main()
