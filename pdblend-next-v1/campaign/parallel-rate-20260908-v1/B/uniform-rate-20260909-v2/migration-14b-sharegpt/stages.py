"""Build fresh B14B qualification inputs only after the measured predecessor."""
import argparse
import copy
import json
from pathlib import Path
import bootstrap as b
import power_selftest as p
import qualify_fixed as q
import restore
HERE=p.HERE
PARENT=p.ROOT/'B/distributed-14b-v1/pdb-p12-release-001/binding.json'
PROFILE=p.ROOT/'B/distributed-14b-v1/frequency2100-registered-001/profiles.development.json'
MODEL=p.ROOT/'A/uniform-rate-20260909-v1/model-manifest.json'
COMMON=p.ROOT/'common/execution-until-complete-v1/run.py'


def source_files():
    result=p.source_check()
    for source in HERE.glob('*.py'):result[str(source)]=p.sha(source)
    for source in [PARENT,PROFILE,MODEL,COMMON,restore.HELPER,restore.POWER,restore.ENERGY]:
        result[str(source)]=p.sha(source)
    return result


def fixed(bootstrap_ref,out):
    assert not out.exists();restore.audit(bootstrap_ref)
    boot=b.checked(bootstrap_ref);parent=b.checked(p.ref(PARENT))
    profile=p.ref(PROFILE);shape=q.shapes(b.checked(profile))
    assert len(shape)==41 and {x[0] for x in shape}=={900,1500,2100}
    template=p.ref(parent['configs']['sharegpt']);cfg=b.checked(template)
    assert cfg['capacity_integration_v1'] is False and cfg['max_service_frequency_mhz']==2100
    assert cfg['idle_domain_reacquire_v1'] is True and cfg['idle_domain_reacquire_timeout_s']==1.5
    assert cfg['request_timeout_s']==120
    out.mkdir(parents=True)
    source_template=template
    cfg=copy.deepcopy(cfg)
    cfg.update(slo_ttft_s=5.,slo_tpot_s=.15)
    b.save(out/'sharegpt-template.json',cfg)
    template=p.ref(out/'sharegpt-template.json')
    oracle=dict(schema='fresh-B14B-cancellation-input-v1',cases=[dict(prompt_length=128,
        prompt=([9707,1879,13]*43)[:128])],native_restoration=bootstrap_ref)
    b.save(out/'cancellation-input.json',oracle)
    files=source_files();files.update(boot['files'])
    for ref in (bootstrap_ref,source_template,template,boot['ordinary'],p.ref(out/'cancellation-input.json')):
        files[ref['path']]=ref['sha256']
    spec=dict(schema='migration-B-fixed14B-qualification-spec-v1',bootstrap=bootstrap_ref,profile=profile,
        stream=p.ref(HERE/'stream.py'),model_manifest=p.ref(MODEL),numerical_reference=p.ref(out/'cancellation-input.json'),
        common_executor=p.ref(COMMON),config_templates={'sharegpt':template},source_config_template=source_template,authorized_slo=dict(ttft_s=5.,tpot_s=.15),shapes=[list(x) for x in shape],files=files,
        all_mixed_tp1_shapes_fresh_observation_required=True,old_node_qualifications_inherited=False)
    b.save(out/'spec.json',spec);q.validate(spec)
    return p.ref(out/'spec.json')


def idle(previous_ref,out):
    import verify_fixed
    prior=verify_fixed.verify(previous_ref)
    assert prior['native_shape_cases']==82 and prior['v3_cancellations']==2 and prior['node']=='B'
    assert not out.exists();out.mkdir(parents=True)
    previous=b.checked(previous_ref);binding=b.checked(prior['binding'])
    cfg=p.read(binding['configs']['sharegpt'])
    assert cfg['idle_domain_reacquire_v1'] is True and cfg['idle_domain_reacquire_timeout_s']==1.5
    files=source_files();files.update(previous['files']);files.update(previous['source_files'])
    files[previous_ref['path']]=previous_ref['sha256']
    spec=dict(schema='migration-B-fixed14B-idle-spec-v1',previous_qualification=previous_ref,
        common_executor=p.ref(COMMON),idle_timeout_s=cfg['idle_domain_reacquire_timeout_s'],files=files)
    b.save(out/'spec.json',spec);return p.ref(out/'spec.json')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('fixed','idle'));ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    print(json.dumps(globals()[a.stage](p.ref(a.input),a.out)))

if __name__=='__main__':main()
