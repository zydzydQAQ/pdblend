"""Exactly one separately authorized technical correction, bound to retained failure."""
import importlib.util,json,os,uuid
from pathlib import Path
import common as c
AUTH=c.ROOT/'technical-authorization.json'

def verify_evidence(a,claims,live):
    c.require(a['schema']==2 and a['attempt_name']=='B32B-temporal-observation-attempt-003','only predeclared third namespace')
    c.verify_files(a['files']);c.require(set(map(str,claims))==set(a['prior_claims']) and len(a['prior_claims'])==2,'unexpected/missing execution claim; no further retry')
    previous=c.read(a['previous_status']);verified=c.read(a['capture_verification']);binding=c.read(a['restored_binding']);claim=c.read(a['prior_claims'][1])
    c.require(claim['sha256']==a['files'][a['previous_spec']] and previous['complete'] and previous['all_original_restored'] and previous['measurement_valid'] and not previous['errors'],'previous observation/restoration not genuinely terminal')
    c.require(verified['capture_verified'] is True and verified['operation_measurement_valid'] is True and verified['original_exact_gate_passed'] is False and verified['hardware_correctness_proven'] is False,'previous capture/correctness boundary changed')
    c.require(verified['input_files'][a['previous_status']]==a['files'][a['previous_status']] and previous['restored_binding_sha256']==a['files'][a['restored_binding']],'capture/current fresh binding linkage differs')
    c.require(binding['configs']=={} and binding['output_correctness_verified'] is False and binding['correctness_gate_required_before_performance'] is True,'fresh bootstrap cannot claim correctness')
    c.require(a['files'][a['scope_manifest']]==c.OBSERVATION_SCOPE_SHA,'wrong new observer declaration')
    for pid in (c.read(a['prior_claims'][0])['pid'],claim['pid'],previous['pid'],previous['child_pid']):c.require(not live(pid),'previous diagnostic process remains')
    return a

def verify():
    a=c.read(AUTH)
    return verify_evidence(a,c.ROOT.parent.glob('B32B-temporal-observation-execution-*/execution-once.json'),c.pid_live)

def compact_spec(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')

def spec_preflight(path,hook=None):
    """The exact engine loader reads the exact future bytes; no writer/tensor/GPU."""
    hook=Path(hook or c.CANDIDATE/'pdblend_diagnostics.py');expected=c.read(c.CANDIDATE/'manifest.json')['files']['pdblend_diagnostics.py'];c.require(c.sha(hook)==expected,'engine hook source differs')
    spec=importlib.util.spec_from_file_location('actual_diagnostic_loader_'+uuid.uuid4().hex,hook);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    keys=('PDBLEND_DIAGNOSTIC_SPEC','PDBLEND_DIAGNOSTIC_SPEC_SHA256');old={k:os.environ.get(k) for k in keys}
    try:
        os.environ[keys[0]]=str(path);os.environ[keys[1]]=c.sha(path);s=m.state()
        c.require(s is not None and s.writer is None and s.error is None and s.spec==c.read(path) and s.spec_sha==c.sha(path) and s.ids==frozenset(s.spec['request_ids']),'actual engine diagnostic loader rejected prepared bytes')
        c.require(m.MAX_BYTES==16384 and m.MAX_RECORDS==14 and m.STEPS==tuple(range(28,35)),'original capture caps changed')
        return dict(passed=True,bytes=Path(path).stat().st_size,spec_sha256=c.sha(path),hook_sha256=c.sha(hook),writer_created=False,hardware_actions=False)
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
