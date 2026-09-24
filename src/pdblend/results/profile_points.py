"""Flatten measured profile points, independently of experiment attempts.

The CSV is an index, not a qualification certificate. It never infers missing
model/system identity, upgrades raw measurements, or combines baseline fits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .journal import read_json

FIELDS = ('point_id','system','model_id','tp','pp','role','frequency_mhz','batch',
          'input_tokens','context_tokens','output_tokens','phase','repeat','samples',
          'seconds','prefill_s','iteration_s','power_w','measurement_s','timing_scope',
          'status','formal_eligible','source_sha256','image_digest','model_hash','tokenizer_hash',
          'raw_path','raw_sha256','sample_path','sample_sha256','qualification_id')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def _value(*sources, names):
    for source in sources:
        if isinstance(source,dict):
            for name in names:
                value=source.get(name)
                if type(value) in (str,int,float,bool):return value
    return ''


def flatten(data,path,checksum):
    binding=data.get('binding',{})
    identity=binding.get('identity',{})
    environment=binding.get('environment',data.get('environment',{}))
    metadata=data.get('metadata',{})
    base=dict(raw_path=str(path),raw_sha256=checksum,formal_eligible=False)
    for key,names in dict(system=('system',),model_id=('model_id',),tp=('tp',),pp=('pp',),
        source_sha256=('source_hash','source_revision','source_sha256'),image_digest=('image_digest',),
        model_hash=('model_hash',),tokenizer_hash=('tokenizer_hash',)).items():
        base[key]=_value(identity,data,metadata,environment,names=names)
    if not base['model_id'] and isinstance(data.get('model'),str):
        base['model_id']=Path(data['model']).name

    def row(point, phase, locator, sample=None):
        sample=sample or point
        summary=sample.get('summary',sample)
        result=dict(base,phase=phase,status='measured_unqualified')
        aliases=dict(role=('role',),frequency_mhz=('frequency_mhz','freq_mhz'),batch=('batch','batch_size'),
            input_tokens=('input_tokens','max_input_tokens'),context_tokens=('effective_context_tokens','context_tokens','max_context_tokens'),
            output_tokens=('output_tokens',),repeat=('repeat',),samples=('samples','runs'),
            seconds=('seconds',),prefill_s=('prefill_s','ttft_seconds'),iteration_s=('iteration_s','step_seconds'),
            power_w=('power_w','mixed_power_w'),measurement_s=('window_s','duration_s','measure_s'),
            timing_scope=('timing_scope','measurement_scope'),sample_path=('samples_file','source_profile_path'),
            sample_sha256=('samples_sha256','source_profile_sha256','sample_sha256'))
        for key,names in aliases.items():result[key]=_value(summary,sample,point,names=names)
        if not result['seconds'] and type(summary.get('stage_latency_ms')) in (float,int):
            result['seconds']=summary['stage_latency_ms']/1000
        if not result['seconds'] and type(summary.get('minimum_ms')) in (float,int):
            result['seconds']=summary['minimum_ms']/1000
        result['qualification_id']=_value(sample.get('qualification',{}),names=('epoch_id',))
        result['point_id']=hashlib.sha256((str(path)+'#'+locator).encode()).hexdigest()[:24]
        return result

    if isinstance(data.get('training'),dict) or isinstance(data.get('holdout'),dict):
        for phase in ('training','holdout'):
            for key,entry in data.get(phase,{}).items():
                for i,sample in enumerate(entry.get('repeats',[])):
                    yield row(entry['point'],phase,f'{phase}/{key}/{i}',sample)
    for role in ('prefill','decode','mixed','transfer','static'):
        points=data.get(role)
        if not isinstance(points,list):continue
        for i,point in enumerate(points):
            if not isinstance(point,dict):continue
            point=dict(point,role=role)
            phase='holdout' if data.get('holdout_independent') else 'unspecified'
            repeats=point.get('repeats')
            if isinstance(repeats,list) and repeats and all(isinstance(r,dict) for r in repeats):
                for j,sample in enumerate(repeats):yield row(point,phase,f'{role}/{i}/{j}',dict(sample,repeat=j))
            else:yield row(point,phase,f'{role}/{i}')
    if data.get('system') in ('distserve','dynamollm','ecoserve'):
        for i,point in enumerate(data.get('points',[])):
            if isinstance(point,dict):yield row(point,'unspecified',f'points/{i}')
        for i,point in enumerate(data.get('rows',[])):
            if not isinstance(point,dict):continue
            if isinstance(point.get('points'),list):
                for j,stage in enumerate(point['points']):
                    yield row(dict(point,**stage),'unspecified',f'rows/{i}/{j}')
            elif 'minimum_ms' in point:
                yield row(dict(point,role='prefill',batch=1,frequency_mhz=metadata.get('frequency_mhz')),
                          'minimum_of_five',f'rows/{i}')


def export(root, output=None):
    root=Path(root).resolve();output=Path(output or root/'profile_points.csv')
    rows,errors,seen=[],[],set()
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not Path(directory,d).is_symlink() and d not in
                 ('sources','source','compiler-cache','__pycache__','.pytest_cache')]
        for name in files:
            # Published aggregates and raw profiles only; individual sample
            # events remain references rather than duplicated CSV measures.
            logical_name=name[:-3] if name.endswith('.gz') else name
            if not (logical_name in ('raw.json','profile.json') or logical_name.endswith('profile.json')
                    or logical_name.endswith('.csv.manifest.json')):continue
            path=Path(directory,name)
            if path.is_symlink():continue
            stat=path.stat();inode=(stat.st_dev,stat.st_ino)
            if inode in seen:continue
            seen.add(inode)
            try:
                data=read_json(path)
                if not isinstance(data,dict):continue
                rows.extend(flatten(data,path,sha(path)))
            except (ValueError,OSError,KeyError,TypeError) as exc:
                errors.append(dict(path=str(path),error=str(exc)))
    # Historical rows whose raw was intentionally pruned stay as tombstones.
    if output.exists():
        with output.open(newline='') as handle:
            previous=list(csv.DictReader(handle))
        current={row['point_id'] for row in rows}
        pruned={row['point_id'] for row in previous if row.get('status')=='raw_pruned'}
        for row in rows:
            if row['point_id'] in pruned:
                row.update(status='raw_pruned',formal_eligible=False)
        rows.extend(row for row in previous if row['point_id'] not in current and row.get('status')=='raw_pruned')
    temporary=output.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    temporary.replace(output)
    return dict(rows=len(rows),output=str(output),errors=errors)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv);report=export(args.root,args.out)
    print(json.dumps(report,indent=2))
    return int(bool(report['errors']))


if __name__=='__main__':raise SystemExit(main())
