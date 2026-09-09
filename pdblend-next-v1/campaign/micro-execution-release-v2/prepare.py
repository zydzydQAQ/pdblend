"""Prepare an owner execution record after actual main and scale completion.

This program never runs a GPU candidate. It reuses the existing read-only
terminal verifier and the candidate's own release validator.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
ADAPTER=CAMPAIGN/'service-dvfs-off-execution-v2/adapter.py'
ADAPTER_SHA='b9761c8c61d138ec70483f7f4f822215978230c086ac9e270563651f55eb9dae'
MODELS={
 '32b':dict(hostname='iZwz9i5bte3xkpmcoes3t2Z',ablation='B32B-service-dvfs-off-v2',
  candidate='B32B-batch16-short512-candidate-v2',manifest='0d8788a2c6a683f020d5d050994563edd0f1e75bde2e9e036ad81c1bb047a102'),
 '14b':dict(hostname='iZwz92bdfqihqp38tekqjyZ',ablation='A14B-service-dvfs-off-v2',
  candidate='A14B-batch19-13-context-candidate-v1',manifest='25feddd47cade9b73e05221487934975fdbc91d83bd8ae414dec7f099eab3823'),
 '7b':dict(hostname='iZwz9gfq11hx1sbob59yrgZ',ablation='C7B-service-dvfs-off-v2',
  candidate='C7B-batch16-context-candidate-v3',manifest='a4370d5aa7919f8ee401dfded7d674953e631bfc20339dbd2f1439d73c85abc4')}


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text())


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def binding(model):
    setting=MODELS[model]
    require(socket.gethostname()==setting['hostname'],'must run on the actual model host, not a result mirror')
    require(sha(ADAPTER)==ADAPTER_SHA,'terminal verifier changed')
    manifest=read(ROOT/'manifest.json')
    for name,digest in manifest['files'].items():require(sha(ROOT/name)==digest,'release helper changed: '+name)
    adapter=load(ADAPTER,'_micro_release_adapter')
    b=adapter.bind(CAMPAIGN/setting['ablation'])
    b.package_check()
    candidate=CAMPAIGN/setting['candidate']
    require(sha(candidate/'manifest.json')==setting['manifest'],'reviewed micro candidate changed')
    candidate_manifest=read(candidate/'manifest.json')
    for name,digest in candidate_manifest['files'].items():require(sha(candidate/name)==digest,'micro source changed: '+name)
    require(not (candidate/'status.json').exists() and not (candidate/'outer-http.jsonl').exists(),
            'existing micro attempt retained; this candidate cannot be repeated in place')
    capacity=load(candidate/'capacity.py','_micro_release_capacity')
    return setting,adapter,b,candidate,capacity


def prepare(model,output):
    setting,adapter,b,candidate,capacity=binding(model)
    output=Path(output).resolve()
    require(not output.exists(),'existing execution record preserved')
    require(output.is_relative_to(ROOT/'records'),'new owner records must stay under this helper records directory')
    # Verifies both original phase ledgers, checkpoint chains and all artifacts;
    # also requires the actual source bridge, queue and child to be gone.
    evidence=adapter.terminal_evidence(b)
    require(evidence['main_completed']==60 and evidence['scale_completed']==36,
            'this helper preserves main/scale priority and only releases after all96 cells')
    phase_refs={}
    for name,digest in evidence['files'].items():
        path=Path(name)
        if path.parent==b.ORIGINAL.ROOT/'invocations':
            status=read(path);phase=status['selected_phase']
            require(phase in ('main','scale') and phase not in phase_refs,'ambiguous terminal phase evidence')
            require(status.get('selected_phase_execution_complete') is True,'actual whole phase is incomplete')
            phase_refs[phase]=dict(path=name,sha256=digest)
    require(set(phase_refs)=={'main','scale'},'both actual terminal invocations required')
    protocol=read(CAMPAIGN/'deadline-24h-v1/protocol.json')
    record=dict(authorized_by='root',execute_once=True,candidate_manifest_sha256=setting['manifest'],
        expires_s=protocol['deadline_s'],phase_terminal_evidence=phase_refs,
        allow_incomplete_phase_termination=False,incomplete_phase_reason=None,
        created_s=time.time(),actual_hostname=socket.gethostname(),prepared_by=str(ROOT/'prepare.py'),
        prepared_by_sha256=sha(ROOT/'prepare.py'),terminal_evidence=evidence,
        gpu_started_by_this_tool=False,scope='owner coordination under the existing user authorization; not a new user approval')
    # The candidate's unchanged validator checks this exact JSON, including
    # the720-second work/cleanup admission reserve and the original deadline.
    with tempfile.TemporaryDirectory(prefix='micro-release-validation-') as temp:
        draft=Path(temp)/'record.json';draft.write_text(json.dumps(record,allow_nan=False)+'\n')
        capacity.validate_execution_release(draft,candidate)
    adapter.file_refs(evidence['files'])
    require(not adapter.source_processes(b.ORIGINAL.ROOT),'source execution restarted during record preparation')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(record,handle,indent=2,allow_nan=False);handle.write('\n')
    return dict(record_path=str(output),record_sha256=sha(output),candidate=str(candidate),
                gpu_started=False,execute_command_requires_own_candidate_gates=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.prepare:
        require(args.out is not None,'an independent record output path is required')
        result=prepare(args.model,args.out)
    else:
        _,adapter,b,candidate,_=binding(args.model)
        result=dict(package_valid=True,actual_hostname=socket.gethostname(),model=args.model,
            candidate=str(candidate),source_processes=adapter.source_processes(b.ORIGINAL.ROOT),
            record_created=False,gpu_started=False,
            phase_completion_not_claimed=True)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
