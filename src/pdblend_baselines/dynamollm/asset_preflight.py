"""CPU-only preflight for incremental native profile or loaded-transition jobs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from .deployment import sha
from .loaded_drain import validate
from .predictor import model_identity
from .profile_v1 import measurement_points
from .transition_probe_v1 import placement


def check(config):
    identity=model_identity(config['model_path'])
    if identity['model']!=config['model_id']:
        raise ValueError('incremental job model identity differs')
    for path,digest in config['immutable_inputs'].items():
        if sha(path)!=digest:raise ValueError('incremental input changed: '+path)
    if config.get('dependencies_manifest'):
        manifest=Path(config['dependencies_manifest']);deps=json.loads(manifest.read_text())
        if deps.get('distribution_version')!='3.2.2' or importlib.metadata.version('PuLP')!='3.2.2':
            raise ValueError('required pinned PuLP distribution differs')
        for name,row in deps['files'].items():
            if sha(manifest.parent/name)!=row['sha256']:raise ValueError('CPU solver bytes differ: '+name)
        from .policy import Configuration,shard_milp
        choices=shard_milp([Configuration(1,1,1,100),Configuration(2,1,3,150)],4,5)
        if sum(choice.tp*n for choice,n in choices)!=4:
            raise ValueError('actual pinned CBC solver preflight failed')
    if config['kind']=='profile':
        points=measurement_points(SimpleNamespace(points_file=config['points_file'],tp=config['tp']),identity['model'])
        if len(config['gpus'])!=config['tp']:raise ValueError('exact TP-sized profile group required')
        batches=sorted({p['batch'] for p in points})
        detail=dict(points=len(points),frequency_bins=sorted({p['frequency_mhz'] for p in points}),
                    batch_values=batches,deferred_batches_above=max(batches),
                    holdout_repeats=1,training_repeats=3)
        if config.get('sampling_epoch_root'):
            from .profile_epochs import DynamoEpochs
            from pdblend.profile.sampling_epochs import SamplingEpochs
            import inspect
            cohort=json.loads((Path(config['sampling_epoch_root'])/'cohort.json').read_text())
            if (cohort.get('cohort_id')!=config['sampling_cohort']
                    or config['sampling_member'] not in cohort['members']
                    or cohort.get('member_gpu_counts',{}).get(config['sampling_member'])!=config['tp']
                    or sum(cohort['member_gpu_counts'].values())!=cohort.get('gpu_budget')
                    or 'probe_callback' not in inspect.signature(SamplingEpochs).parameters):
                raise ValueError('Dynamo profile sampling cohort or callback API differs')
            detail.update(sampling_cohort=cohort['cohort_id'],sampling_member=config['sampling_member'],
                coordinator_members=cohort['members'],require_fresh_runtime_qualification=True,
                own_probe_module=DynamoEpochs.__module__)
    elif config['kind']=='loaded_transition':
        source,target=placement(identity['model'],config['gpus'],19000)
        validate(config['drain_input'],config['drain_output'],config['drain_batch'])
        detail=dict(source_tp=source['tp'],target_tp=target['tp'],
                    workload_envelope={k:config[k] for k in ('drain_input','drain_output','drain_batch')})
    else:raise ValueError('unknown incremental GPU task')
    return dict(ready=True,model_id=identity['model'],kind=config['kind'],detail=detail,
                gpu_started=False,formal_eligible=False,source_snapshot=config['source_snapshot'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    print(json.dumps(check(json.loads(parser.parse_args(argv).config.read_text()))))


if __name__=='__main__':main()
