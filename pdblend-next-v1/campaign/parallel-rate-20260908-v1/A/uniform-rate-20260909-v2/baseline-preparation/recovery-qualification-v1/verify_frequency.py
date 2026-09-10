"""Join independently rechecked candidate evidence and a distinct real cache export."""
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
D=HERE.parent/'retention-recovery-v1'

def physical_binding(derived,original):
    new=p.checked(derived);old=p.checked(original)
    p.need({k:v for k,v in new.items() if k!='files'}=={k:v for k,v in old.items() if k!='files'},'composed qualification changed physical binding')
    p.need(all(new['files'].get(name)==digest for name,digest in old['files'].items()),'prior physical/source closure not retained')
    for name,digest in new['files'].items():p.need(p.sha(name)==digest,'composed source/input changed')

def value(binding_ref,profile_ref,original_binding,prior_out,retention_out):
    physical_binding(binding_ref,original_binding)
    prior_ref=p.ref(Path(prior_out)/'status.json');retention_ref=p.ref(Path(retention_out)/'status.json')
    prior=p.checked(prior_ref);retained=p.checked(retention_ref)
    p.need(prior['binding']==retained['binding']==original_binding and prior['profile']==retained['profile']==profile_ref,'different physical/profile evidence domains')
    p.need(prior['finished_s']<retained['started_s'],'retention must follow completed original cleanup')
    return dict(schema='composed-legacy-frequency-and-retention-evidence-v1',binding=binding_ref,profile=profile_ref,
        original_binding=original_binding,prior_frequency=prior_ref,retention=retention_ref,
        retained_weights=retained['retained_weights'],topology=retained['topology'],
        original_failure_preserved=True,physical_load_repeated=False,
        prior_verifier=p.ref(D/'verify_prior.py'),retention_verifier=p.ref(D/'verify_retention.py'))

def create(out,binding_ref,profile_ref,original_binding,prior_out,retention_out):
    out=Path(out);p.need(not out.exists(),'fresh composition required')
    p.save(out/'status.json',value(binding_ref,profile_ref,original_binding,prior_out,retention_out))

def verify(out,binding_ref,profile_ref):
    out=Path(out);reference=p.ref(out/'status.json');saved=p.checked(reference)
    prior_out=Path(saved['prior_frequency']['path']).parent;retention_out=Path(saved['retention']['path']).parent
    p.need(saved==value(binding_ref,profile_ref,saved['original_binding'],prior_out,retention_out),'composition declaration differs from actual independent evidence')
    prior=p.load(saved['prior_verifier'],'composed_original_frequency').verify(prior_out,saved['original_binding'],profile_ref)
    retention=p.load(saved['retention_verifier'],'composed_real_retention').verify(retention_out,saved['original_binding'],profile_ref)
    p.need(prior['passed'] and prior['independently_recomputed'] and prior['retained_weights_qualified'] is False,'prior candidate evidence incomplete or false cache claim')
    p.need(retention['passed'] and retention['independently_recomputed'],'actual independent retention proof failed')
    files={**prior['files'],**retention['files']}
    for item in (reference,binding_ref,profile_ref,saved['original_binding'],saved['prior_frequency'],saved['retention'],saved['prior_verifier'],saved['retention_verifier'],p.ref(__file__)):
        files[item['path']]=item['sha256']
    return dict(passed=True,independently_recomputed=True,cases=prior['cases'],requests=prior['requests'],
        native_idle_wakeup_cycles=prior['native_idle_wakeup_cycles'],frequency_domain_mhz=prior['frequency_domain_mhz'],
        loaded_tolerance_mhz=prior['loaded_tolerance_mhz'],historical_profile_costs_recalibrated=False,
        original_frequency_failed_status_preserved=saved['prior_frequency'],retention_measurement=saved['retention'],
        composition=reference,frequency_evidence=prior,retention_evidence=retention,files=files)
