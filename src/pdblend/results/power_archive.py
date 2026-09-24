"""Compact sampler archive with exact samples and explicit source epochs.

The public sampler arrays and acquisition cadence are unchanged. New native
jobs opt into this serializer after stopping the sampler. Immutable historical
power.json files continue through the same reader without conversion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .journal import CompactJournal, iter_journal, read_json

SCHEMA='pdblend-power-v1'
FIXED=('gpus','mode','source_id','field_id','scope_id','value_type')
ARRAYS=('samples','frequency_samples','utilization_samples','power_metadata')


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_power_archive(path,value):
    path=Path(path);raw=path.with_name(path.stem+'.samples.jsonl.gz')
    if path.exists() or raw.exists():raise FileExistsError('refusing to overwrite power evidence')
    manifest={key:item for key,item in value.items() if key not in ARRAYS}
    identities,lookup=[],{}
    counts={key:0 for key in ARRAYS if key in value}
    with CompactJournal(raw) as stream:
        for name in ARRAYS:
            for sample in value.get(name,[]):
                if name=='power_metadata':
                    fixed={key:sample[key] for key in FIXED if key in sample}
                    key=json.dumps(fixed,sort_keys=True,allow_nan=False)
                    if key not in lookup:
                        lookup[key]=len(identities);identities.append(fixed)
                    variable={key:item for key,item in sample.items() if key not in FIXED}
                    stream.write(dict(kind=name,source_epoch=lookup[key],values=variable))
                else:
                    stream.write(dict(kind=name,values=sample))
                counts[name]+=1
    manifest.update(schema=SCHEMA,raw_path=raw.name,raw_sha256=_sha(raw),counts=counts,
                    power_source_epochs=identities)
    path.write_text(json.dumps(manifest,sort_keys=True,allow_nan=False)+'\n')
    return manifest


def read_power_archive(path):
    path=Path(path);manifest=read_json(path)
    if manifest.get('schema')!=SCHEMA:return manifest
    raw=path.parent/manifest['raw_path']
    if raw.parent.resolve()!=path.parent.resolve() or _sha(raw)!=manifest['raw_sha256']:
        raise ValueError('power sample binding differs')
    value={key:item for key,item in manifest.items() if key not in
           ('schema','raw_path','raw_sha256','counts','power_source_epochs')}
    value.update({key:[] for key in manifest['counts']})
    for row in iter_journal(raw):
        kind=row['kind']
        if kind not in value or kind not in ARRAYS:raise ValueError('unexpected power sample kind')
        sample=row['values']
        if kind=='power_metadata':
            epoch=row['source_epoch']
            if type(epoch) is not int or not 0<=epoch<len(manifest['power_source_epochs']):
                raise ValueError('power source epoch differs')
            sample={**manifest['power_source_epochs'][epoch],**sample}
        value[kind].append(sample)
    if any(len(value[key])!=count for key,count in manifest['counts'].items()):
        raise ValueError('power sample count differs')
    return value
