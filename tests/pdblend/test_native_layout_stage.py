from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from pdblend.profile.collection import native_layout_stage as stage
from pdblend.profile.collection.native_timing_plan import binding
from pdblend_runtime.probe import NativeSpec
from test_native_timing_replay_v2 import collected_v2,collected,plans
from test_native_timing_replay import put


def resident(x):
    report=deepcopy(x.v2_complete);report.pop('physical_cleanup')
    report.pop('actual_engine_starts');report.pop('engine_loads');report['hardware_executed']=False
    for row in report['final_drains']:
        for key in ('drain','state'):row[key].update(total_blocks=1024,free_blocks=1024,reserved_blocks=0)
    specs=[NativeSpec(**row['spec']) for row in report['actual_launch']]
    fleet={s.instance_id:SimpleNamespace(spec=s,alive=lambda:True,
        process=SimpleNamespace(pid=x.v2_complete['actual_engine_starts'][s.instance_id][-1]['pid']),
        events=x.v2_complete['actual_engine_starts'][s.instance_id]) for s in specs}
    return report,specs,fleet


def test_live_stage_replays_same_tp2_raw_without_worker_success_or_empty_gpu_claim(collected_v2):
    x=collected_v2;report,specs,fleet=resident(x)
    execution=x.attempt/'execution.json';saved=execution.read_bytes();execution.unlink()
    ref=stage.capture_resident_timing(report,input_manifest_ref=x.inputs_ref,attempt_manifest_ref=binding(x.attempt/'manifest.json'),
        specs=specs,fleet=fleet,out=x.root/'resident-stage.json')
    result=stage.replay_resident_timing(ref)
    assert result['supported_fit']==x.v2_fitted and result['resident_boundary']['live_instances']==4
    assert not result['physical_cleanup_verified'] and not result['queue_terminal_verified'] and not result['formal_eligible']
    assert not execution.exists()
    # Final lifetime verification is a separate existing queue+cleanup replay.
    execution.write_bytes(saved)
    from pdblend.profile.collection.native_timing_replay import capture_evidence
    final=capture_evidence(x.attempt,x.queue,x.tmp/'final-evidence.json')
    verified=stage.verify_final_timing(ref,final)
    assert verified['supported_fit']==result['supported_fit']


def test_resident_stage_rejects_false_release_claim_stale_drains_or_dead_rank(collected_v2):
    x=collected_v2;report,specs,fleet=resident(x)
    dead=fleet[specs[0].instance_id];dead.alive=lambda:False
    with pytest.raises(ValueError,match='lost an actual engine'):
        stage.capture_resident_timing(report,input_manifest_ref=x.inputs_ref,attempt_manifest_ref=binding(x.attempt/'manifest.json'),
            specs=specs,fleet=fleet,out=x.root/'dead.json')
    dead.alive=lambda:True
    ref=stage.capture_resident_timing(report,input_manifest_ref=x.inputs_ref,attempt_manifest_ref=binding(x.attempt/'manifest.json'),
        specs=specs,fleet=fleet,out=x.root/'resident-stage.json')
    original=json.loads(Path(ref['path']).read_text())
    for index,kind in enumerate(('claim','pid','inventory','stage_time','drain')):
        value=deepcopy(original)
        if kind=='claim':value['physical_cleanup_verified']=True
        elif kind=='pid':value['live_instances'][0]['process_pid']+=1
        elif kind=='inventory':value['live_instances'].pop()
        elif kind=='stage_time':value['captured_s']=1.
        else:
            changed=deepcopy(report);changed['final_drains'][0]['drain']['generation']+=1
            value['report']=put(x.tmp/'wrong-generation.json',changed)
        altered=put(x.tmp/f'stage-bad-{index}.json',value)
        with pytest.raises((ValueError,RuntimeError)):stage.replay_resident_timing(altered)
