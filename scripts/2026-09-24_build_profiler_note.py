#!/usr/bin/env python3
"""Reproduce the profiler note's numbers from checksum-bound calibration data."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/profiler-method-2026-09-24'
sys.path.insert(0, str(ROOT / 'src'))
from pdblend.profile.query.versions import load_version


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registry = ROOT / 'results/2026-09-23/calibration-versions-v1/registry.json'
    version = next(v for v in json.loads(registry.read_text())['versions']
                   if v['model_id'] == 'Qwen2.5-7B-Instruct')
    loaded = load_version(registry, version['version_id'], system='pdblend',
                          model_id=version['model_id'], tp=4, pp=1, usage='development')
    model = loaded.model
    f, s, b, c, output, horizon = 2100, 2048, 12, 2112, 128, 60
    assert model.decode_supported(b, c, f) and model.decode_power_supported(b, c, f)
    tp, pp = model.prefill_seconds(s, f), model.prefill_power_w(s, f)
    td, pd = model.step_seconds(b, c, f), model.decode_power_w(b, f, ctx=c)
    marginal = model.prefill_marginal_seconds(s, f)
    # Choose an illustrative load that closes Little's law at B=12, N=1.
    rate = b / (output * td + b * marginal)
    busy = rate * marginal
    wait = busy * marginal / (2 * (1 - busy))
    mixed_tpot = td / (1 - busy)
    mixed_power = busy * pp + (1 - busy) * pd
    assert busy < .7 and math.isclose(rate * output * mixed_tpot, b, rel_tol=1e-12)

    table = version['decode_power_domains'][str(f)]['nodes']
    interpolation = []
    for batch in (8, 16):
        nodes = sorted((x for x in table if x['batch'] == batch), key=lambda x: x['context_min'])
        lo, hi = next((lo, hi) for lo, hi in zip(nodes, nodes[1:])
                      if lo['context_max'] < c < hi['context_min'])
        w = (c - lo['context_max']) / (hi['context_min'] - lo['context_max'])
        power = lo['power_w'] + w * (hi['power_w'] - lo['power_w'])
        assert math.isclose(power, model.decode_power_w(batch, f, ctx=c), rel_tol=1e-12)
        interpolation.append(dict(batch=batch, left=lo, right=hi, weight=w, power_w=power))
    assert math.isclose(pd, sum(x['power_w'] for x in interpolation) / 2, rel_tol=1e-12)

    raw_path = Path(version['original_inputs']['original_raw']['path'])
    raw = json.loads(raw_path.read_text())
    p_observed = next(r for r in raw['prefill'] if r['freq_mhz'] == f and r['input_tokens'] == s)
    d_observed = next(r for r in raw['decode'] if r['freq_mhz'] == f and r['batch'] == b
                      and r['context_tokens'] == 2048)
    d_prediction = model.step_seconds(b, d_observed['effective_context_tokens'], f)
    comparisons = [dict(kind='prefill_time', observed_s=p_observed['seconds'], predicted_s=tp,
                        error=abs(tp / p_observed['seconds'] - 1)),
                   dict(kind='decode_time', observed_s=d_observed['step_seconds'], predicted_s=d_prediction,
                        effective_context=d_observed['effective_context_tokens'],
                        error=abs(d_prediction / d_observed['step_seconds'] - 1))]
    candidate = Path(version['evidence']['power_candidate']['path'])
    core = json.loads(candidate.read_text())
    result = dict(
        version_id=version['version_id'], registry_sha256=sha(registry),
        input=dict(model=version['model_id'], tp=4, pp=1, frequency_mhz=f,
                   prompt_tokens=s, batch=b, effective_context=c, mean_output_tokens=output),
        prefill=dict(time_s=tp, power_w=pp, energy_estimate_j=tp*pp, marginal_time_s=marginal),
        decode=dict(step_s=td, power_w=pd, energy_per_token_j=td*pd/b),
        interpolation=interpolation,
        mixed=dict(illustrative_load=True, measured_mixed_result=False, instances=1,
                   rate_rps=rate, horizon_s=horizon, busy_fraction=busy, queue_wait_s=wait,
                   tpot_s=mixed_tpot, ttft_proxy_s=wait+tp+mixed_tpot,
                   power_w=mixed_power, window_energy_j=horizon*mixed_power,
                   fixed_point_batch=rate*output*mixed_tpot),
        holdout_comparisons=comparisons, qualification=loaded.qualification,
        coefficients={k:core[k][str(f)] for k in ('prefill_time','prefill_power','decode_overrides')},
        source_files={str(p.relative_to(ROOT)):sha(p) for p in
                      (registry,candidate,raw_path,ROOT/'src/pdblend/profile/query/model.py',
                       ROOT/'src/pdblend/profile/query/decode.py',ROOT/'src/pdblend/profile/query/power_table.py',
                       ROOT/'src/pdblend/planner/pool.py')})
    data = OUT / 'data'
    data.mkdir(parents=True, exist_ok=True)
    (data/'example.json').write_text(json.dumps(result, indent=2)+'\n')
    (data/'profile-snapshot.json').write_bytes(candidate.read_bytes())
    (data/'holdout-excerpts.json').write_text(json.dumps(dict(
        source_path=str(raw_path),source_sha256=sha(raw_path),
        prefill=p_observed,decode=d_observed),indent=2)+'\n')
    values = dict(Ptime=f'{tp*1000:.3f}', Ppower=f'{pp:.3f}', Penergy=f'{tp*pp:.3f}',
                  Dtime=f'{td*1000:.3f}', Dpower=f'{pd:.3f}', Denergy=f'{td*pd/b:.4f}',
                  Pmarginal=f'{marginal*1000:.3f}', Rate=f'{rate:.4f}', Busy=f'{busy:.4f}',
                  Queue=f'{wait*1000:.3f}', Mtpot=f'{mixed_tpot*1000:.3f}',
                  Mttft=f'{(wait+tp+mixed_tpot)*1000:.3f}', Mpower=f'{mixed_power:.3f}',
                  Menergy=f'{horizon*mixed_power/1000:.3f}',
                  PeObserved=f'{comparisons[0]["observed_s"]*1000:.3f}',
                  PeError=f'{comparisons[0]["error"]*100:.2f}',
                  DeContext=f'{comparisons[1]["effective_context"]:.3f}',
                  DeObserved=f'{comparisons[1]["observed_s"]*1000:.3f}',
                  DePredicted=f'{comparisons[1]["predicted_s"]*1000:.3f}',
                  DeError=f'{comparisons[1]["error"]*100:.2f}')
    for name, row in zip(('Eight','Sixteen'), interpolation):
        values.update({name+'Lo':f'{row["left"]["context_max"]:.4f}'.rstrip('0').rstrip('.'),
                       name+'Hi':f'{row["right"]["context_min"]:.4f}'.rstrip('0').rstrip('.'),
                       name+'Plo':f'{row["left"]["power_w"]:.3f}',
                       name+'Phi':f'{row["right"]["power_w"]:.3f}',
                       name+'Weight':f'{row["weight"]:.6f}',
                       name+'Power':f'{row["power_w"]:.3f}'})
    (data/'numbers.tex').write_text('\n'.join('\\newcommand{\\ex'+k+'}{'+v+'}' for k,v in values.items())+'\n')
    print(json.dumps(dict(version=result['version_id'],prefill=result['prefill'],decode=result['decode'],
                          mixed=result['mixed'],holdout=comparisons),indent=2))
    build = OUT/'build'
    build.mkdir(exist_ok=True)
    for source in ('figures/profiler-process.tex','main.tex'):
        for _ in range(2 if source == 'main.tex' else 1):
            proc=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',
                                 '-file-line-error','-output-directory=build',source],
                                cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (build/(Path(source).stem+'.console.log')).write_text(proc.stdout)
            if proc.returncode:
                print(proc.stdout[-7000:]);raise SystemExit(proc.returncode)
    (OUT/'pdblend-profiler.pdf').write_bytes((build/'main.pdf').read_bytes())
    print(OUT/'pdblend-profiler.pdf')


if __name__ == '__main__':
    main()
