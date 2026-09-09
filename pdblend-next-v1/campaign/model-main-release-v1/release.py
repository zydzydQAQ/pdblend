"""Explicit 7B/14B model-scoped scale release; never a global450 release."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'main-first-barrier-v2'

def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

import hashlib
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sha(PARENT/'barrier.py') == 'f097fe250049113c3b31733bbfb6ed530a03e5042c12b5060306042189e894d4'
assert sha(PARENT/'partition.py') == '36aa52bfc07fc14e25dd48cac3f317b1bcd2966a2da2642c1c5a268ac07f7747'
old = sys.modules.get('partition')
try:
    sys.modules['partition'] = load('model_release_partition', PARENT/'partition.py')
    per_cell = load('model_release_original_barrier', PARENT/'barrier.py')
finally:
    if old is None:
        sys.modules.pop('partition', None)
    else:
        sys.modules['partition'] = old
p = per_cell.p
v1 = per_cell.v1
require, read = v1.require, v1.read
PROTOCOL, DEADLINE = per_cell.PROTOCOL, per_cell.DEADLINE
KIND = 'model-main-release-per-cell'
AUTHORIZED = {
    '14b': ('083e11adc673dae16ce4ccc31946c4b0e7a8ae4019ef3526fb1b447f17788fcc',
            '175282b034b900af1ab183b04b956299be752f236bf8c7393b91c40fe32160f3'),
    '7b': ('f4cc98971fc6117b67057a457b50a7b38960138e5368f8159b2ef079c78f3e91',
           '46b08a7584ef2ed79adc9a9254e82e3646967fca8b2150f590bb8775c23b8d47'),
}

def package_check():
    m = read(HERE/'manifest.json')
    for path,h in m['files'].items():
        require(sha(HERE/path) == h, 'model release source changed: '+path)
    for path,h in m['dependencies'].items():
        require(sha(path) == h, 'frozen release dependency changed: '+path)
    per_cell.package_check()

def release_contract(value, *, expected_model=None, now_s=None):
    now = time.time() if now_s is None else now_s
    model = value.get('model')
    require(model in AUTHORIZED and (expected_model is None or model == expected_model),
            'release does not authorize this model')
    require(value.get('schema') == 1 and value.get('kind') == KIND
        and value.get('protocol_id') == PROTOCOL and value.get('deadline_s') == DEADLINE,
        'model-only release kind/protocol/deadline differs')
    require(set(value.get('models', {})) == {model} and set(value.get('proof_refs', {})) == {model}
        and value.get('baseline_systems') == list(v1.BASELINES)
        and value.get('main_records') == 150 and value.get('baseline_main_records') == 120
        and value.get('pdblend_main_records') == 30 and value.get('coordinator_deep_verification') is True
        and value.get('global_release') is False, 'requires exactly this model150, never global450')
    proof = value['models'][model]
    require(per_cell.proof_contract(proof) == model, 'wrong original host proof')
    ref = value['proof_refs'][model]
    require(ref.get('sha256') == AUTHORIZED[model][0]
        and ref.get('canonical_sha256') == AUTHORIZED[model][1] == v1.digest_json(proof),
        'not the authorized original model proof')
    require(proof['created_s'] <= value['created_s'] <= now < DEADLINE,
        'model release publication/deadline invalid')
    return model

def assemble_release(proof_path, proof_sha, out, *, path_map=None):
    """CPU deep raw review then new immutable model release. No host controls."""
    package_check()
    require(not Path(out).exists(), 'release output must be new')
    require(sha(proof_path) == proof_sha, 'original actual proof SHA changed')
    proof = read(proof_path)
    model = proof.get('model')
    require(model in AUTHORIZED and proof_sha == AUTHORIZED[model][0]
        and v1.digest_json(proof) == AUTHORIZED[model][1], 'unapproved model/proof')
    require(per_cell.verify_model_proof(proof, deep=True, path_map=path_map) == model,
            'original model raw proof failed')
    require(sha(proof_path) == proof_sha, 'proof changed during deep review')
    value = dict(schema=1, kind=KIND, model=model, protocol_id=PROTOCOL, deadline_s=DEADLINE,
        baseline_systems=list(v1.BASELINES), models={model:proof},
        proof_refs={model:dict(path=str(Path(proof_path).resolve()),sha256=proof_sha,
            canonical_sha256=v1.digest_json(proof))},
        created_s=time.time(), coordinator_hostname=socket.gethostname(), coordinator_deep_verification=True,
        main_records=150, baseline_main_records=120, pdblend_main_records=30, global_release=False,
        user_authorization='2026-09-08: 7B and14B may each start SLO scale while32B diagnosis continues.',
        scale_authorization='Only the named model; original same-work scale/reference/identity/deadline/lease gates remain required.')
    release_contract(value, expected_model=model)
    v1.write_new(out,value)
    return dict(release=str(Path(out).resolve()),sha256=sha(out),model=model,kind=KIND,main_records=150,global_release=False)

def verify_release(path, expected_sha256, *, expected_model=None,
                   expected_protocol_id=PROTOCOL, expected_deadline_s=DEADLINE,
                   deep=False, path_maps=None):
    require(v1.valid_sha(expected_sha256) and sha(path) == expected_sha256,
            'explicit pinned model release SHA required')
    require(expected_protocol_id == PROTOCOL and expected_deadline_s == DEADLINE,
            'scale protocol/deadline differs')
    value=read(path)
    model=release_contract(value,expected_model=expected_model)
    if deep:
        per_cell.verify_model_proof(value['models'][model],deep=True,path_map=(path_maps or {}).get(model))
    require(sha(path)==expected_sha256,'model release changed during read')
    return dict(released=True,kind=KIND,model=model,protocol_id=PROTOCOL,deadline_s=DEADLINE,
        main_records=150,global_release=False,
        verified_scope='this_model_raw_files_and_contract' if deep else 'pinned_model_release_sha_and_contract',
        root_release_sha256=expected_sha256,local_deadline_and_lease_still_required=True,per_cell_main_sources=True)

def main():
    q=argparse.ArgumentParser(description=__doc__)
    sub=q.add_subparsers(dest='command',required=True)
    a=sub.add_parser('assemble-release');a.add_argument('--proof',type=Path,required=True)
    a.add_argument('--proof-sha256',required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--path-map',type=Path,help='Explicit original-path to identical local-byte path map for deep review')
    a=sub.add_parser('verify-release');a.add_argument('--release',type=Path,required=True)
    a.add_argument('--sha256',required=True);a.add_argument('--model',choices=sorted(AUTHORIZED),required=True)
    a.add_argument('--deep',action='store_true')
    args=q.parse_args();package_check()
    result=(assemble_release(args.proof,args.proof_sha256,args.out,path_map=read(args.path_map) if args.path_map else None) if args.command=='assemble-release'
        else verify_release(args.release,args.sha256,expected_model=args.model,deep=args.deep))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
