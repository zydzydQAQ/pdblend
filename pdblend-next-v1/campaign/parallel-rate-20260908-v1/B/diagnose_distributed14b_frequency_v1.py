import collections,csv,hashlib,importlib.util,json,pathlib,sys,time,socket
B=pathlib.Path(__file__).resolve().parent;R=B.parent;out=B/'distributed-14b-v1';p=out/'profile-validation-001'
sys.path[:0]=[str(R/'hosts/14b-capacity-p9/src'),str(R/'hosts/14b-capacity-p9'),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.measurement import power_evidence
from ecopadg.measure.power import trapezoid_energy
from ecopadg.metrics import clip_power_window
read=lambda f:json.loads(pathlib.Path(f).read_text())
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
ref=lambda f:dict(path=str(f),sha256=sha(f))
s=read(p/'status.json');assert s['errors']==["ValueError('loaded SM outside original +/-15 MHz command band')"] and s['sampling_error'] is None
power=list(csv.DictReader((p/'power/power.csv').open()));samples=[(float(r['t_s']),[float(r[f'gpu{i}_w']) for i in range(8)]) for r in power];metadata=[json.loads(x) for x in (p/'power/power_metadata.jsonl').read_text().splitlines()];pe=power_evidence(samples,read(p/'power/power_source.json'),metadata);assert pe['power_source_verified']
energy=trapezoid_energy(clip_power_window(samples,s['operation_start_s'],s['operation_end_s'],pad_s=0));assert abs(energy-s['all8_operation_energy_j'])<1e-6
assert s['clock_restore_complete'] and all(x['complete'] and not x['errors'] for x in s['native_cleanup'])
raw=read(p/'gpu6-mid16-2520/raw.json');events=[json.loads(x) for x in (p/'gpu6-mid16-2520/events.jsonl').read_text().splitlines()];events=[e for e in events if e['request_ids']];clocks=list(csv.DictReader((p/'power/clocks.csv').open()));active=[float(x['gpu6_sm_mhz']) for x in clocks if any(e['started_s']<=float(x['t_s'])<=e['finished_s'] for e in events)]
assert len(raw['requests'])==16 and all(x['success'] and len(x['output_token_ids'])==64 for x in raw['requests']) and raw['cleanup']['complete']
v=dict(schema='B14B-frequency-instability-diagnosis-v1',hostname=socket.gethostname(),generated_s=time.time(),status=ref(p/'status.json'),original_measurement_valid_preserved=s['measurement_valid'],raw_energy_measurement_valid=True,scientific_comparison_eligible=False,energy_j=energy,power_evidence=pe,reason='requested2520 mixed D16 native work actually runs2400..2520; original +/-15 qualification fails; no output failure; not permission to relabel2520 as2400 profile',frequency_counts=dict(collections.Counter(active)),min_mhz=min(active),max_mhz=max(active),loaded_samples=len(active),outside_original_band=sum(abs(x-2520)>15 for x in active),work_complete=True,n_requests=16,output_tokens=1024,native_cleanup_complete=True,clock_restore_complete=True,prior_three_passed=s['points'],files={str(f):sha(f) for f in p.rglob('*') if f.is_file()},source=ref(pathlib.Path(__file__)))
with (out/'profile-frequency-diagnosis-001.json').open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(dict(energy_j=energy,range=[min(active),max(active)],outside_original_band=v['outside_original_band'],total=len(active),diagnosis=ref(out/'profile-frequency-diagnosis-001.json'))))
