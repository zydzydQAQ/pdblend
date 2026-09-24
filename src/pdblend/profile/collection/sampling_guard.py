"""Bind resident measurement windows to explicit, checksum-verified epochs."""
import inspect
import json
from pathlib import Path
from pdblend.profile.collection.long_context_collect import digest


async def call(callback,*args):
    value=callback(*args)
    return await value if inspect.isawaitable(value) else value


def snapshot(guard):
    value=guard()
    if (not isinstance(value,dict) or any(not isinstance(value.get(k),str) or not value[k]
            for k in ('epoch_id','qualification_path','qualification_sha256','layout_sha256'))):
        raise ValueError('measurement epoch qualification binding missing')
    path=Path(value['qualification_path'])
    if not path.is_file() or digest(path)!=value['qualification_sha256']:raise ValueError('epoch qualification checksum differs')
    receipt=json.loads(path.read_text())
    if (receipt.get('complete') is not True or receipt.get('cross_job') is not True or
            not (receipt.get('passed') is True or receipt.get('fallback')=='serial_cohort')):
        raise ValueError('epoch needs real isolated/parallel qualification')
    return dict(value)


def save_binding(out,binding):
    path=Path(out)/'samples'/('qualification-'+binding['qualification_sha256']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    data=Path(binding['qualification_path']).read_bytes()
    if path.exists() and path.read_bytes()!=data:raise ValueError('saved epoch qualifier differs')
    if not path.exists():path.write_bytes(data)
    return dict(samples_file=str(path.relative_to(out)),samples_sha256=binding['qualification_sha256'],
        epoch_id=binding['epoch_id'],layout_sha256=binding['layout_sha256'])


def unchanged(guard,before):
    after=snapshot(guard)
    if after!=before:raise ValueError('measurement epoch changed during window; sample not accepted')


def static_guard(profiler):
    """Existing fixed-cohort qualification; never claim dynamic membership."""
    binding=profiler.raw.get('external_interference',{})
    return lambda:dict(epoch_id='fixed-cohort',qualification_path=str(profiler.out_dir/binding.get('samples_file','')),
        qualification_sha256=binding.get('samples_sha256'),layout_sha256=binding.get('samples_sha256'))
