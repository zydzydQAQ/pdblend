import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import adapter
import ablation_queue as q


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def phase_fixture(tmp_path):
    protocol = tmp_path / 'original-protocol.json'
    write(protocol, dict(created_s=1000, deadline_s=87400))
    spec = dict(model='32b', deadline_scope=dict(path=str(protocol), sha256=adapter.sha(protocol)),
                execution_budget=dict(startup_allowance_s=60, restore_allowance_s=120))
    write(tmp_path / 'runspec.json', spec)
    write(tmp_path / 'freeze.json', dict(prepared_s=84000))
    return SimpleNamespace(ROOT=tmp_path), spec


def test_independent_phase_keeps_original_absolute_end(tmp_path):
    b, spec = phase_fixture(tmp_path)
    first = q.phase_record(b, now=85000, create=True)
    second = q.phase_record(b, now=86000, create=True)
    assert first == second and first['deadline_s'] == 87400
    assert first['phase'] == 'service_dvfs_off'
    assert not (tmp_path / 'phase-ledgers').exists()


@pytest.mark.parametrize('field,value', [('deadline_s', 99999), ('phase', 'main'), ('started_s', 86000),
                                        ('model', '7b'), ('deadline_scope_sha256', 'changed')])
def test_phase_cannot_reset_or_borrow_main_clock(tmp_path, field, value):
    b, spec = phase_fixture(tmp_path)
    record = q.phase_record(b, now=85000, create=True)
    record[field] = value
    write(tmp_path/'phase-ledger.json', record)
    with pytest.raises(RuntimeError):
        q.phase_record(b, now=85000)


def test_no_window_can_fit_near_absolute_deadline(tmp_path):
    b, spec = phase_fixture(tmp_path)
    clock = q.phase_record(b, now=86000)
    shared = adapter.load(Path(__file__).resolve().parent.parent/'fixed-window-queue-v1/queue.py', '_test_shared')
    assert shared.cell_limits(spec, clock, now=87000) is None
    limits = shared.cell_limits(spec, clock, now=86800)
    assert limits['phase'] == 'service_dvfs_off' and limits['restore_deadline_s'] == 87400


@pytest.mark.parametrize('status,complete,count', [('preflight',False,0),('cell:any',False,0),
    ('failed',False,1),('stopped_by_request',True,1),('finished',True,1)])
def test_active_failed_or_partial_batch_not_terminal(tmp_path,status,complete,count):
    b=SimpleNamespace(ROOT=tmp_path)
    write(tmp_path/'invocations/000001.json',dict(selected_phase='main',phase=status,complete=complete,
        checkpointed_cells=count,baseline_preservation_verified=True,finished_s=100))
    with pytest.raises(RuntimeError):
        adapter.terminal_invocation(b,'main',dict(completed=count,phase_execution_complete=False))


def test_clean_deadline_terminal_remains_partial_not_complete_matrix(tmp_path):
    b=SimpleNamespace(ROOT=tmp_path)
    write(tmp_path/'invocations/000001.json',dict(selected_phase='main',phase='stopped_by_deadline',complete=True,
        checkpointed_cells=20,baseline_preservation_verified=True,finished_s=100))
    _,value=adapter.terminal_invocation(b,'main',dict(completed=20,phase_execution_complete=False))
    assert value['checkpointed_cells']==20


def test_original_source_process_blocks_even_before_child(tmp_path):
    source=tmp_path/'source';proc=tmp_path/'proc';path=proc/'42/cmdline';path.parent.mkdir(parents=True)
    path.write_bytes(('python\0'+str(source/'run.py')+'\0--phase\0run\0').encode())
    assert adapter.source_processes(source,proc)[0]['pid']==42


def test_terminal_gate_blocks_active_source_before_reading_evidence(tmp_path,monkeypatch):
    original=SimpleNamespace(ROOT=tmp_path)
    monkeypatch.setattr(adapter,'source_processes',lambda root:[{'pid':42}])
    b=SimpleNamespace(ORIGINAL=original,SOURCE_QUEUE=None)
    with pytest.raises(RuntimeError,match='still active'):
        adapter.terminal_evidence(b)


def test_terminal_gate_never_creates_missing_phase_ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(adapter,'source_processes',lambda root:[])
    write(tmp_path/'runspec.json',dict(model='32b',deadline_scope={'path':str(tmp_path/'protocol.json')}))
    b=SimpleNamespace(ORIGINAL=SimpleNamespace(ROOT=tmp_path),SOURCE_QUEUE=None)
    with pytest.raises(RuntimeError,match='ledger missing'):
        adapter.terminal_evidence(b)
    assert not (tmp_path/'phase-ledgers').exists()


def test_completed_status_still_readable_after_global_deadline(tmp_path):
    b,_=phase_fixture(tmp_path)
    record=q.phase_record(b,now=85000,create=True)
    assert q.phase_record(b,now=90000)==record


def test_no_new_ledger_created_after_global_deadline(tmp_path):
    b,_=phase_fixture(tmp_path)
    q.phase_record(b,now=90000,create=True)
    assert not (tmp_path/'phase-ledger.json').exists()


def test_phase_cannot_bind_changed_prerequisite(tmp_path):
    b,_=phase_fixture(tmp_path)
    q.phase_record(b,now=85000,create=True)
    write(tmp_path/'freeze.json',dict(prepared_s=84001))
    with pytest.raises(RuntimeError,match='clock changed'):
        q.phase_record(b,now=85000)


def test_selection_uses_original_scale1_references_not_good_results():
    main=[dict(phase='main',cell_id=str(i),slo_attainment=0) for i in range(18)]
    scale=[dict(phase='scale',reuse_main_cell_id=str(i)) for i in range(18)]
    assert adapter.selected_sources(dict(cells=main+scale))==main
    with pytest.raises(RuntimeError):
        adapter.selected_sources(dict(cells=main+scale[:-1]))


def test_original_row_work_and_slo_unchanged():
    source=dict(cell_id='orig',phase='main',part='main',sequence=22,trace='a',trace_sha256='sha',
        n_requests=100,seed=701,slo_scale=1,slo_ttft_s=1,slo_tpot_s=.1)
    before=copy.deepcopy(source)
    candidate=adapter.candidate_row(source,1,Path('/new/config'),'newhash')
    assert source==before
    for key in ('trace','trace_sha256','n_requests','seed','slo_scale','slo_ttft_s','slo_tpot_s'):
        assert candidate[key]==source[key]


def test_unfavorable_but_valid_receipt_not_filtered():
    shared=adapter.load(Path(__file__).resolve().parent.parent/'fixed-window-queue-v1/queue.py','_test_receipt')
    row=dict(n_requests=10,trace_sha256='x',slo_scale=1,slo_ttft_s=1,slo_tpot_s=.1)
    limits=dict(latest_arrival_epoch_s=10,cell_execution_deadline_s=430,restore_deadline_s=550)
    summary=dict(measurement_valid=True,measurement_schema=3,measurement_window_protocol=shared.PROTOCOL,
        fixed_window_valid=True,n_expected=10,trace_sha256='x',slo_scale=1,measurement_end_s=400,
        post_measurement_cleanup=dict(cleanup_complete=True),slo_attainment=0,work_complete=False,
        fixed_window=dict(arrival_epoch_s=10,arrival_window_s=300,effective_slo_s=dict(ttft=1,tpot=.1)))
    receipt=dict(screen_valid=True,outer_cleanup=dict(complete=True),summary=summary,finished_s=410)
    assert shared.valid_receipt(receipt,row,limits) is summary
    receipt['outer_cleanup']['complete']=False
    with pytest.raises(ValueError):shared.valid_receipt(receipt,row,limits)


@pytest.mark.parametrize('field,value', [('schema',1),('completed_s',float('nan')),('completed_s',201),
    ('completed_s',90),('measurement_valid',False),('work_complete',True),('slo_attainment',1),
    ('good_requests',10),('energy_j',99)])
def test_checkpoint_header_must_match_verified_summary(field,value):
    summary=dict(work_complete=False,slo_attainment=0.,good_requests=0,energy_j=100)
    record=dict(schema=2,completed_s=150,limits=dict(issued_s=100),measurement_valid=True,**summary)
    record[field]=value
    with pytest.raises(RuntimeError):q.checkpoint_header(record,summary,dict(deadline_s=200))


def test_checkpoint_header_accepts_valid_failure_without_selection():
    summary=dict(work_complete=False,slo_attainment=0.,good_requests=0,energy_j=100)
    record=dict(schema=2,completed_s=150,limits=dict(issued_s=100),measurement_valid=True,**summary)
    q.checkpoint_header(record,summary,dict(deadline_s=200))


def test_checkpoint_cannot_omit_metric_from_both_header_and_summary():
    summary=dict(work_complete=False,slo_attainment=0.,good_requests=0)
    record=dict(schema=2,completed_s=150,limits=dict(issued_s=100),measurement_valid=True,**summary)
    with pytest.raises(RuntimeError,match='metric header'):
        q.checkpoint_header(record,summary,dict(deadline_s=200))


def queue_fixture(tmp_path,monkeypatch):
    b,spec=phase_fixture(tmp_path)
    shared=adapter.load(Path(__file__).resolve().parent.parent/'fixed-window-queue-v1/queue.py','_test_resume')
    rows=[dict(sequence=i+1,phase='service_dvfs_off',cell_id='new-'+str(i),source_main_cell_id='old-'+str(i),
               n_requests=10,trace_sha256='trace',slo_scale=1,slo_ttft_s=1,slo_tpot_s=.1) for i in range(18)]
    spec['cells']=rows;write(tmp_path/'runspec.json',spec)
    refs={r['source_main_cell_id']:{'checkpoint_sha256':'reference'} for r in rows}
    write(tmp_path/'freeze.json',dict(prepared_s=84000,terminal_evidence={'references':refs}))
    write(tmp_path/'package-manifest.json',dict(schema=1))
    b.SOURCE_QUEUE=shared;b.read=adapter.read;b.sha=adapter.sha;b.package_check=lambda:spec
    monkeypatch.setattr(q.time,'time',lambda:85000)
    clock=q.phase_record(b,create=True)
    limits=shared.cell_limits(spec,clock,now=85000)
    cid=rows[0]['cell_id'];summary=dict(measurement_valid=True,measurement_schema=3,
        measurement_window_protocol=shared.PROTOCOL,fixed_window_valid=True,n_expected=10,
        trace_sha256='trace',slo_scale=1,measurement_end_s=85000,
        post_measurement_cleanup=dict(cleanup_complete=True),slo_attainment=0,work_complete=False,
        good_requests=0,energy_j=100,
        fixed_window=dict(arrival_epoch_s=85000,arrival_window_s=300,effective_slo_s=dict(ttft=1,tpot=.1)))
    write(tmp_path/'cells'/cid/'summary.json',summary)
    write(tmp_path/'operations'/cid/'evidence.json',{'retained':True})
    write(tmp_path/'identities'/(cid+'.json'),{'actual':True})
    write(tmp_path/'receipts'/(cid+'.json'),dict(screen_valid=True,outer_cleanup=dict(complete=True),
        summary=summary,finished_s=85000))
    shared.checkpoint(b,rows[0],limits,None,refs[rows[0]['source_main_cell_id']])
    return b,rows


def test_resume_only_skips_complete_immutable_prefix_including_bad_slo(tmp_path,monkeypatch):
    b,rows=queue_fixture(tmp_path,monkeypatch)
    result=q.inspect_queue(b)
    assert result['completed']==1 and result['remaining']==17
    assert result['records'][0]['work_complete'] is False and result['records'][0]['slo_attainment']==0


def test_resume_rejects_changed_artifact(tmp_path,monkeypatch):
    b,rows=queue_fixture(tmp_path,monkeypatch)
    write(tmp_path/'operations'/rows[0]['cell_id']/'evidence.json',{'retained':False})
    with pytest.raises(RuntimeError,match='dependency changed'):
        q.inspect_queue(b)


def test_resume_never_overwrites_uncheckpointed_attempt(tmp_path,monkeypatch):
    b,rows=queue_fixture(tmp_path,monkeypatch)
    (tmp_path/'operations'/rows[1]['cell_id']).mkdir()
    with pytest.raises(RuntimeError,match='uncheckpointed'):
        q.inspect_queue(b)


def test_resume_rejects_metric_header_tampering(tmp_path,monkeypatch):
    b,rows=queue_fixture(tmp_path,monkeypatch)
    path=next((tmp_path/'checkpoints/service_dvfs_off').glob('*.json'))
    record=adapter.read(path);record['energy_j']=1;write(path,record)
    with pytest.raises(RuntimeError,match='metric header'):
        q.inspect_queue(b)
