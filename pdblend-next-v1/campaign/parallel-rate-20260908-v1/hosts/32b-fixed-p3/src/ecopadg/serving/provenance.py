"""Bind a formal artifact freeze to live engine processes, not just files."""
import asyncio
import json
from pathlib import Path


MODEL_ROOT=Path('/root/workspace/models/Qwen2.5-14B-Instruct')


def verify_model(raw,freeze):
    if raw.get('model')!='/models/Qwen2.5-14B-Instruct':
        raise ValueError('running engine differs from the fixed 14B model')
    index=MODEL_ROOT/'model.safetensors.index.json'
    try:
        weights=set(json.loads(index.read_text())['weight_map'].values())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('fixed model shard index unavailable') from exc
    required={str((MODEL_ROOT/name).resolve()) for name in weights|
              {'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}}
    if not weights or not required<=set(freeze['groups'].get('model',[])):
        raise ValueError('formal freeze must include the fixed model weights and tokenizer')


def verify_engine_source(raw,instance_id,config,freeze):
    serving_root=Path(__file__).resolve().parent
    required={p:freeze['files'][p] for p in freeze['groups']['source']
              if Path(p).resolve().parent==serving_root and p.endswith('.py')}
    local={str(p.resolve()) for p in Path(__file__).parent.glob('*.py')}
    if set(required)!=local:
        raise ValueError('formal freeze must contain every serving source file')
    if (raw.get('instance_id')!=instance_id or raw.get('tp')!=config['tp']
            or raw.get('engine_version')!='0.9.2' or raw.get('dtype')!='bfloat16'
            or raw.get('max_model_len')!=8192
            or raw.get('cuda_visible_devices')!=','.join(map(str,config['gpus']))
            or raw.get('source_files_at_import')!=required):
        raise ValueError('running engine differs from frozen implementation/configuration: '+instance_id)
    verify_model(raw,freeze)


async def verify_live_engines(backend,freeze):
    expected_image=freeze['identities']['engine_image']
    records=[]
    for instance_id,config in tuple(backend.instances.items()):
        if not instance_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in instance_id):
            raise ValueError('invalid experimental instance identity')
        raw=await backend.json(instance_id,'/provenance')
        verify_engine_source(raw,instance_id,config,freeze)
        process=await asyncio.create_subprocess_exec('docker','inspect','--format','{{.Image}}',
            'pdb-v2-'+instance_id,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try: stdout,stderr=await asyncio.wait_for(process.communicate(),10)
        except BaseException:
            if process.returncode is None: process.kill();await process.wait()
            raise
        if process.returncode or stdout.decode().strip()!=expected_image:
            raise ValueError('running engine image differs from immutable freeze: '+instance_id)
        records.append(dict(raw,image_id=expected_image))
    return records
