"""Hash the actual serving policy and its frozen source/profile inputs."""
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def identity(binding,dataset):
    config_path=Path(binding['configs'][dataset]);config=read(config_path)
    profile_path=Path(config['profiles']) if config.get('profiles') else None
    host_path=Path(binding['host_release'])/'manifest.json'
    if profile_path is None and config['strategy']!='mixed':
        raise ValueError('profile-dependent strategy lacks a profile')
    for path in (config_path,host_path,*([profile_path] if profile_path else [])):
        if binding['files'].get(str(path))!=sha(path):
            raise ValueError('serving source/config/profile is not frozen in binding: '+str(path))
    host=read(host_path)
    for relative,expected in host['files'].items():
        if sha(host_path.parent/relative)!=expected:
            raise ValueError('serving source manifest file differs: '+relative)
    excluded={'journal','port','profiles','host_source_release','controller_source_release',
        'engine_source_release','candidate_label','profile_compatibility','transfer_evidence',
        'frequency_evidence','interconnect'}
    policy={k:v for k,v in config.items() if k not in excluded and k!='instances'}
    policy['instances']=sorted([dict(tp=i['tp'],gpus=sorted(i['gpus']),role=i['role'])
        for i in config['instances']],key=lambda i:i['gpus'])
    source=host.get('common_controller_sha256',digest(host['files']))
    result=dict(controller_source_sha256=source,host_manifest_sha256=sha(host_path),
        profile_sha256=sha(profile_path) if profile_path else None,policy_sha256=digest(policy),
        configured_gpu_count=len({g for i in config['instances'] for g in i['gpus']}),
        configured_instance_count=len(config['instances']),configured_tp_sizes=[i['tp'] for i in config['instances']],
        frozen_policy_verified=True)
    result['version_id']=digest(dict(source=source,profile=result['profile_sha256'],policy=result['policy_sha256']))
    return result

def original_identity(point):
    source=point['executed_source'];path=source['binding_path']
    if sha(path)!=source['binding_sha256']:raise ValueError('original baseline binding changed')
    return identity(read(path),point['dataset'])
