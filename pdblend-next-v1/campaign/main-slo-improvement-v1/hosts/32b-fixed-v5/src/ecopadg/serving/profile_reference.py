"""Read an independent, explicitly prepared empty-GPU power reference.

This is derived calibration evidence, not a formal serving run. Physical
occupancy comes from the completed preparation, never the profile's topology
complement. Original preparation and operator measurements remain unchanged.
"""
import hashlib
import json
import math
from pathlib import Path

from ecopadg.measure.power import trapezoid_mean_power
from ecopadg.metrics import clip_power_window
from .measurement import power_evidence


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _layout(rows):
    instances={};occupied=set()
    for row in rows:
        name=row.get('instance_id',row.get('id'));tp=row.get('tp');gpus=row.get('gpus',[])
        if (not isinstance(name,str) or not name or name in instances
                or type(tp) is not int or tp not in (1,2,4,8) or len(gpus)!=tp
                or any(type(g) is not int or not 0<=g<8 for g in gpus)
                or len(set(gpus))!=tp or occupied.intersection(gpus)):
            raise ValueError('invalid or overlapping prepared physical layout')
        instances[name]=dict(tp=tp,gpus=list(gpus));occupied.update(gpus)
    return instances,occupied


def measured_unallocated_reference(spec):
    """Return maximum per-empty-GPU idle-window watts, evidence and hashes."""
    artifacts={}
    def read(path):
        path=Path(path).resolve();data=path.read_bytes()
        artifacts[str(path)]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    raw_path=Path(spec['raw']).resolve();prepare_path=Path(spec['preparation']).resolve()
    raw=read(raw_path);preparation=read(prepare_path)
    for item in (raw,preparation):
        if (item.get('complete') is not True or item.get('errors') or item.get('sampling_error')
                or item.get('cleanup_errors') or item.get('cleanup_complete') is False):
            raise ValueError('unallocated reference requires complete successful measurements')
    manifest=preparation.get('manifest',{})
    if any(key not in manifest for key in ('add','keep','remove','image')):
        raise ValueError('unallocated reference requires explicit complete preparation layout')
    _,previous=_layout(manifest['remove']+manifest['keep'])
    if previous!=set(range(8)):
        raise ValueError('preparation does not establish complete prior eight-GPU occupancy')
    resident,resident_gpus=_layout(manifest['add']+manifest['keep'])
    added,_=_layout(manifest['add'])
    prepared,_=_layout([item.get('spec',{}) for item in preparation.get('instances',[])])
    if prepared!=added:
        raise ValueError('completed startup instances differ from requested preparation')
    free=sorted(set(range(8))-resident_gpus)
    if not free:raise ValueError('preparation leaves no unallocated GPU')
    topology=raw.get('topology',{})
    if not topology:raise ValueError('missing operator topology')
    used={}
    for row in topology.values():
        name=row.get('id',row.get('instance_id'))
        if name not in resident or resident[name]!=dict(tp=row.get('tp'),gpus=row.get('gpus')):
            raise ValueError('operator topology does not match prepared physical instances')
        used[name]=resident[name]
    engine_path=Path(__file__).with_name('engine.py').resolve()
    engine_hash=hashlib.sha256(engine_path.read_bytes()).hexdigest()
    image=manifest['image'];records=raw.get('engine_provenance',[]);seen=set()
    if not isinstance(image,str) or not image.startswith('sha256:') or len(image)!=71:
        raise ValueError('preparation requires an immutable engine image')
    for record in records:
        name=record.get('instance_id');identity=used.get(name)
        hashes=[value for path,value in record.get('source_files_at_import',{}).items()
                if path.endswith('/serving/engine.py')]
        if (identity is None or name in seen or record.get('image_id')!=image
                or record.get('model')!='/models/Qwen2.5-14B-Instruct'
                or record.get('engine_version')!='0.9.2' or hashes!=[engine_hash]
                or record.get('tp')!=identity['tp']
                or record.get('cuda_visible_devices')!=','.join(map(str,identity['gpus']))):
            raise ValueError('missing, stale or mismatched operator engine provenance')
        seen.add(name)
    if seen!=set(used):raise ValueError('missing operator engine provenance')
    power=raw.get('power_samples',[])
    if (len(power)<2 or any(not _finite(t) or len(ws)!=8 or
            any(not _finite(w) or w<0 for w in ws) for t,ws in power)
            or any(b[0]<=a[0] for a,b in zip(power,power[1:]))):
        raise ValueError('invalid eight-GPU power samples')
    provenance=power_evidence(power,raw.get('power_source'),raw.get('power_metadata'))
    if not provenance['power_source_verified']:
        raise ValueError('unallocated reference requires verified instantaneous eight-GPU power')
    finished=preparation.get('finished_s');started=preparation.get('started_s')
    if not _finite(finished) or not _finite(started) or finished<started:
        raise ValueError('invalid preparation time bounds')
    windows=set()
    for run in raw.get('runs',[]):
        frequency=run.get('frequency_mhz');start=run.get('idle_start_s');end=run.get('idle_end_s')
        if (type(frequency) is not int or frequency not in (900,1500,2100,2520)
                or not _finite(start) or not _finite(end) or not finished<=start<end
                or start<power[0][0] or end>power[-1][0]):
            raise ValueError('idle window is invalid, precedes preparation or exceeds power samples')
        windows.add((frequency,start,end))
    if {f for f,_,_ in windows}!={900,1500,2100,2520}:
        raise ValueError('unallocated reference requires idle windows at all four frequencies')
    measurements=[]
    for frequency,start,end in sorted(windows):
        rows=clip_power_window(power,start,end,pad_s=0)
        values={str(g):trapezoid_mean_power([(t,[ws[g]]) for t,ws in rows]) for g in free}
        if any(not _finite(w) or w<0 for w in values.values()):
            raise ValueError('invalid integrated unallocated power')
        measurements.append(dict(frequency_mhz=frequency,started_s=start,finished_s=end,
                                 watts_by_gpu=values))
    watts=max(w for window in measurements for w in window['watts_by_gpu'].values())
    helper=Path(__file__).resolve()
    artifacts[str(helper)]=hashlib.sha256(helper.read_bytes()).hexdigest()
    artifacts[str(engine_path)]=engine_hash
    evidence=dict(method='maximum per-unallocated-GPU mean power across measured idle windows; no model margin',
        scope='independent calibration reference; not a formal serving run',formal_eligible=False,
        watts=watts,free_gpus=free,resident_gpus=sorted(resident_gpus),resident_instances=resident,
        source_path=str(raw_path),source_sha256=artifacts[str(raw_path)],
        preparation_path=str(prepare_path),preparation_sha256=artifacts[str(prepare_path)],
        preparation_finished_s=finished,engine_image=image,engine_source_sha256=engine_hash,
        windows=measurements,power_source=dict(raw['power_source']),power_evidence=provenance,
        boundary='each recorded idle window, entirely after preparation and within raw samples',
        measurement_changed=False)
    return watts,evidence,artifacts
