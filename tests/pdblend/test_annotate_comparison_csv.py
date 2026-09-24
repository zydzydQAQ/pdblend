"""Host annotation publication uses tiny synthetic CSV/receipt fixtures only."""
import csv
import fcntl
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/2026-09-24_annotate_comparison_csv.py"
spec = importlib.util.spec_from_file_location("annotate_comparison_csv", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"point_sha256": "point-digest"}))
    binding = dict(receipt_path=str(receipt), receipt_sha256=mod.sha(receipt.read_bytes()),
                   point_id="7b-pdblend-longbench-x1-seed701", point_sha256="point-digest",
                   system="pdblend", run_id="profile-saturation-repaired-v1", revision="d4fe")
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(json.dumps(dict(
        schema="pdblend-cold-writer-service-overlap-review/v1",
        receipt={"path": str(receipt), "sha256": binding["receipt_sha256"]},
        writer_launch_recorded_s=17.0, writer_first_publish_s=36.5,
        service_start_s=10.0, service_end_s=160.0,
        launch_to_first_publish_overlap_with_service_s=19.5,
        exact_cpu_export_start_instrumented=False,
    )))
    columns = dict(zip(mod.COLUMNS, (
        "observed_launch_to_first_publish_overlap", 19.5, 17.0, 36.5, mod.SEMANTICS,
        "unknown", str(diagnostic), mod.sha(diagnostic.read_bytes()),
    )))
    registry = dict(schema=mod.SCHEMA, annotations=[dict(binding=binding, columns=columns)],
                    default_columns={k: "not_annotated" if k == mod.COLUMNS[0] else "" for k in mod.COLUMNS})
    fields = list(mod.BINDING) + ["energy_service_j", "ttft_p99_s", "slo_pass", "free_text"]
    rows = [dict(binding, energy_service_j="00123.4500", ttft_p99_s="3.7892749309539795",
                 slo_pass="True", free_text='Chinese 中文, comma\nsecond "line"'),
            dict(binding, receipt_path="old-receipt", receipt_sha256="old", point_id="baseline",
                 system="mixed", energy_service_j="", ttft_p99_s="0.0", slo_pass="False", free_text=" ")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    csv_path = tmp_path / "compare.csv"
    csv_path.write_bytes(stream.getvalue().encode())
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    lock = tmp_path / "owner.lock"
    lock.touch()
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"leases": {}}))
    Path(str(queue) + ".lock").touch()
    return csv_path, registry_path, lock, queue, registry, rows


def execute(f, *, apply=False):
    csv_path, registry_path, lock, queue, _, _ = f
    return mod.annotate(csv_path, registry_path, mod.sha(registry_path.read_bytes()), lock, queue, apply=apply)


def test_preserves_all_old_cells_and_adds_only_annotation_columns(tmp_path):
    f = fixture(tmp_path)
    csv_path, _, _, queue, _, old = f
    queue_before = queue.read_bytes()
    report = execute(f, apply=True)
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(), newline="")))
    assert len(rows) == len(old)
    assert [{k: r[k] for k in old[0]} for r in rows] == old
    assert set(rows[0]) - set(old[0]) == set(mod.COLUMNS)
    assert rows[0]["writer_startup_overlap_s"] == "19.5"
    assert rows[0]["writer_startup_causal_impact"] == "unknown"
    assert rows[1]["writer_startup_overlap_status"] == "not_annotated"
    assert rows[1]["writer_startup_overlap_s"] == ""
    assert queue.read_bytes() == queue_before
    assert report["existing_nonannotation_cells_preserved"] == len(old) * len(old[0])


@pytest.mark.parametrize("field", mod.BINDING)
def test_wrong_binding_rejected_without_csv_change(tmp_path, field):
    f = fixture(tmp_path)
    original = f[0].read_bytes()
    f[4]["annotations"][0]["binding"][field] = "wrong"
    f[1].write_text(json.dumps(f[4]))
    with pytest.raises((ValueError, FileNotFoundError)):
        execute(f, apply=True)
    assert f[0].read_bytes() == original


def test_repeated_application_is_byte_identical(tmp_path):
    f = fixture(tmp_path)
    assert execute(f, apply=True)["changed"] is True
    first = f[0].read_bytes()
    report = execute(f, apply=True)
    assert report["changed"] is False
    assert report["input_csv_sha256"] == report["output_csv_sha256"]
    assert f[0].read_bytes() == first


def test_dry_run_never_writes_csv(tmp_path):
    f = fixture(tmp_path)
    original = f[0].read_bytes()
    assert execute(f)["applied"] is False
    assert f[0].read_bytes() == original


def test_active_lease_rejects_annotation(tmp_path):
    f = fixture(tmp_path)
    original = f[0].read_bytes()
    f[3].write_text(json.dumps({"leases": {"another-run": {"status": "active"}}}))
    with pytest.raises(RuntimeError, match="active GPU lease"):
        execute(f, apply=True)
    assert f[0].read_bytes() == original


def test_live_writer_lock_rejects_annotation(tmp_path):
    f = fixture(tmp_path)
    original = f[0].read_bytes()
    with f[2].open("r+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            execute(f, apply=True)
    assert f[0].read_bytes() == original


def test_conflicting_annotation_is_not_overwritten(tmp_path):
    f = fixture(tmp_path)
    execute(f, apply=True)
    f[0].write_bytes(f[0].read_bytes().replace(b"19.5", b"19.6"))
    original = f[0].read_bytes()
    with pytest.raises(ValueError, match="conflicting existing annotation"):
        execute(f, apply=True)
    assert f[0].read_bytes() == original


def test_unapproved_metric_column_rejected(tmp_path):
    f = fixture(tmp_path)
    f[4]["annotations"][0]["columns"]["energy_service_j"] = 0
    f[1].write_text(json.dumps(f[4]))
    with pytest.raises(ValueError, match="eight approved columns"):
        execute(f, apply=True)


def test_timing_cannot_be_relabelled_as_cpu_busy_time(tmp_path):
    f = fixture(tmp_path)
    f[4]["annotations"][0]["columns"]["writer_startup_interval_semantics"] = "cpu_busy"
    f[1].write_text(json.dumps(f[4]))
    with pytest.raises(ValueError, match="verified diagnostic"):
        execute(f, apply=True)


def test_uncoordinated_csv_change_prevents_atomic_replace(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    original_prepare = mod.prepare_csv
    concurrent_bytes = b"another writer changed this file\n"

    def race(data, registry):
        result = original_prepare(data, registry)
        f[0].write_bytes(concurrent_bytes)
        return result

    monkeypatch.setattr(mod, "prepare_csv", race)
    with pytest.raises(RuntimeError, match="CSV changed"):
        execute(f, apply=True)
    assert f[0].read_bytes() == concurrent_bytes
    assert not list(tmp_path.glob("compare.csv.annotation-*"))


def test_missing_lock_is_not_created(tmp_path):
    f = fixture(tmp_path)
    f[2].unlink()
    with pytest.raises(FileNotFoundError):
        execute(f, apply=True)
    assert not f[2].exists()


def update_receipt(f, payload):
    binding = f[4]["annotations"][0]["binding"]
    receipt = Path(binding["receipt_path"])
    receipt.write_text(json.dumps(payload))
    binding["receipt_sha256"] = mod.sha(receipt.read_bytes())
    rows = list(csv.DictReader(io.StringIO(f[0].read_text(), newline="")))
    rows[0]["receipt_sha256"] = binding["receipt_sha256"]
    with f[0].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return binding


def recovered_fixture(tmp_path, change=None):
    """Small synthetic diagnostics: a failed drain with independently valid service."""
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=mod.sha(path.read_bytes()))

    directory = tmp_path/'session/windows/failed'
    source = write(tmp_path/'source.json', {'source_sha256': 'rev'})
    uuids = ['GPU-'+str(i) for i in range(8)]
    point = dict(name='dist-failed', system='distserve', run_id='run', revision='rev',
                 source_manifest=source, duration_s=150., slo={'ttft_s': 1., 'tpot_s': .1},
                 trace={'path': str(tmp_path/'trace.json'), 'sha256': 'a'*64},
                 engine_identity={'fleet_gpu_uuids': uuids})
    point_ref = write(directory/'point.json', point)
    point_digest = mod.sha(json.dumps(point, sort_keys=True, separators=(',', ':')).encode())
    method = write(directory/'run/metering-method-failure.json',
                   dict(factory={'sha256': 'm'*64}, gpu_uuids=uuids))
    receipt = write(directory/'receipt.json', dict(point_sha256=point_digest, cleanup_passed=False,
        error='drain500', artifacts={'point.json': point_ref['sha256'], 'run/completion.json': 'c'*64,
                                    'run/metering-method-failure.json': method['sha256']}))
    binding = dict(receipt_path=receipt['path'], receipt_sha256=receipt['sha256'], point_id=point['name'],
                   point_sha256=point_digest, system='distserve', run_id='run', revision='rev')
    bindings = dict(receipt=receipt, point_sha256=point_digest, source_manifest=source, trace=point['trace'])
    stats = dict(samples=2, p50_s=.1, p90_s=.4, p95_s=.4, p99_s=.4, max_s=.4, mean_s=.25)
    native = dict(schema='distserve-native-failed-window-scalar-review/v1', bindings=bindings,
        canonical_modified=False, used_for_ranking=False, canonical_energy_service_j=None,
        canonical_energy_tail_j=None, native_completion={'path':str(directory/'run/completion.json'),'sha256':'c'*64},
        counts=dict(expected_requests=4, recorded_outcomes=4, native_successful_requests=2,
                    native_failed_or_missing_requests=2, native_joint_slo_requests=1, native_joint_slo_rate=.25,
                    all_outcomes_accounted=True),
        missing_outcome_indices=[], invalid_success_timing_indices=[], native_ttft=stats.copy(), native_tpot=stats.copy(),
        native_request_slo_pass=False, client_send_queue_status='unknown_actual_send_and_connector_queue',
        service_started_s=1., service_ended_s=151.,
        native_goodput_finished_window_request_s_lower_bound=1/150,
        native_cohort_goodput_request_s=.005, native_cohort_goodput_token_s=.1,
        client_pre_dispatch_delay={'p99_s':.1}, client_peak_pre_dispatch_outstanding=3)
    service = dict(start_s=1., end_s=151., duration_s=150.,
        power=dict(status='complete', energy_j=80., coverage_fraction=1., minimum_gpu_coverage_fraction=1.,
                   max_gap_s=.1, per_gpu={u:dict(status='complete', integral=10.) for u in uuids}),
        utilization=dict(coverage_fraction=1.))
    power = dict(schema='failed-native-recovered-service-power/v1', bindings=bindings,
        canonical_modified=False, used_for_ranking=False, canonical_energy_service_j=None,
        canonical_energy_tail_j=None, recovered_tail_available=False, outer_drain_complete=False,
        recovered_energy_tail_j=None, recovered_service_available=True,
        power_source_verified=True, power_error_affects_service=False, gpu_uuids=uuids,
        service_start_s=1., service_end_s=151., service=service, recovered_energy_service_j=80.,
        recovered_service_mean_power_w=80/150, recovered_gpu_util_mean_pct=25.,
        session_power_manifest={'path':str(directory.parent.parent/'session-power.json'),'sha256':'p'*64},
        session_power_raw={'path':str(directory.parent.parent/'session-power.samples.jsonl.gz'),'sha256':'r'*64},
        original_method={'path':str(tmp_path/'original_method.py'),'sha256':'m'*64},
        raw_binding_scope='raw SHA stored in original session-power manifest; manifest SHA first captured by this independent review, not included in the failed window receipt')
    if change: change(native, power)
    native_ref = write(tmp_path/'native-review.json', native)
    power['native_scalar_review'] = native_ref
    power_ref = write(tmp_path/'power-review.json', power)
    columns = mod.recovered_columns(native, power, native_ref, power_ref)
    registry = dict(schema=mod.RECOVERED_SCHEMA, annotations=[dict(binding=binding, columns=columns)],
                    default_columns={k:'not_annotated' if k==mod.RECOVERED_COLUMNS[0] else '' for k in mod.RECOVERED_COLUMNS})
    rp = tmp_path/'registry.json'; rp.write_text(json.dumps(registry))
    rows = [dict(binding, status='failed', failure_reason='drain500', energy_service_j='', energy_tail_j='', slo_pass=''),
            dict(binding, receipt_path='historical', receipt_sha256='old', point_id='baseline', status='measured',
                 failure_reason='', energy_service_j='00123.4500', energy_tail_j='0.0', slo_pass='False')]
    cp = tmp_path/'compare.csv'
    with cp.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lock=tmp_path/'owner.lock'; lock.touch(); queue=tmp_path/'queue.json';queue.write_text('{"leases":{}}')
    Path(str(queue)+'.lock').touch()
    return cp,rp,lock,queue,registry,rows


def test_recovery_preserves_failed_canonical_cells_and_is_idempotent(tmp_path):
    f = recovered_fixture(tmp_path)
    original = f[5]
    assert execute(f, apply=True)['changed']
    rows = list(csv.DictReader(io.StringIO(f[0].read_text())))
    assert [{k:r[k] for k in original[0]} for r in rows] == original
    assert rows[0]['recovered_service_energy_j'] == '80.0'
    assert rows[0]['recovered_tail_energy_j'] == ''
    assert rows[0]['recovered_used_for_ranking'] == 'False'
    assert rows[0]['recovered_ttft_samples'] == '2'
    assert 'review_not_failed_receipt' in rows[0]['recovered_power_binding_scope']
    assert rows[1]['recovered_status'] == 'not_annotated'
    assert rows[0]['recovered_gpu0_uuid'] == 'GPU-0'
    assert rows[0]['recovered_gpu0_util_mean_pct'] == ''
    assert rows[0]['recovered_gpu0_util_status'].startswith('unknown_missing_fields:')
    assert rows[0]['recovered_native_goodput_token_s_lower_bound'] == ''
    assert rows[0]['recovered_native_window_goodput_status'] == 'unknown_missing_scalar_request_evidence'
    first=f[0].read_bytes(); assert not execute(f,apply=True)['changed']; assert f[0].read_bytes()==first


def test_recovered_per_gpu_identity_order_and_partial_missing_values():
    uuids = ['GPU-'+str(i) for i in range(8)]
    devices = {uuid: dict(mean_pct=float(i), peak_pct=90., coverage_fraction=1.,
                         max_gap_s=.1, status='complete') for i, uuid in reversed(list(enumerate(uuids)))}
    del devices['GPU-3']['peak_pct']
    power = dict(gpu_uuids=uuids, service=dict(utilization=dict(per_gpu=devices)))
    fields = mod.recovered_device_columns(power)
    assert fields['recovered_gpu7_uuid'] == 'GPU-7'
    assert fields['recovered_gpu7_util_mean_pct'] == 7.
    assert fields['recovered_gpu7_util_status'] == 'complete'
    assert fields['recovered_gpu3_util_mean_pct'] == 3.
    assert fields['recovered_gpu3_util_peak_pct'] == ''
    assert fields['recovered_gpu3_util_status'] == 'unknown_missing_fields:peak_pct'
    devices['other-GPU'] = dict(devices['GPU-0'])
    with pytest.raises(ValueError, match='UUID'):
        mod.recovered_device_columns(power)


@pytest.mark.parametrize('field,value', [('mean_pct', True), ('peak_pct', 101.),
                                       ('coverage_fraction', .9), ('max_gap_s', 1.1)])
def test_recovered_per_gpu_invalid_or_false_complete_rejected(field, value):
    device = dict(mean_pct=25., peak_pct=90., coverage_fraction=1., max_gap_s=.1, status='complete')
    device[field] = value
    power = dict(gpu_uuids=[str(i) for i in range(8)], service=dict(utilization=dict(per_gpu={'0': device})))
    with pytest.raises(ValueError):
        mod.recovered_device_columns(power)


def token_cache_fixture(tmp_path):
    point = dict(slo=dict(ttft_s=1., tpot_s=.1))
    point_file = tmp_path/'point.json'; point_file.write_text(json.dumps(point))
    rows = [dict(idx=0, ok=True, ttft_s=.5, tpot_s=.05, finished_s=150., completion_tokens=20),
            dict(idx=1, ok=True, ttft_s=.5, tpot_s=.05, finished_s=152., completion_tokens=30),
            dict(idx=2, ok=False)]
    cache = tmp_path/'scalars.json'; cache.write_text(json.dumps(rows))
    native = dict(scalar_requests=dict(path=str(cache), sha256=mod.sha(cache.read_bytes())),
        bindings=dict(point_artifact=dict(path=str(point_file), sha256=mod.sha(point_file.read_bytes())),
                      point_sha256=mod.sha(json.dumps(point, sort_keys=True, separators=(',', ':')).encode())),
        service_started_s=1., service_ended_s=151., native_goodput_finished_window_request_s_lower_bound=1/150,
        counts=dict(expected_requests=3, native_successful_requests=2, native_joint_slo_requests=2,
                    native_joint_slo_completion_tokens=50, native_good_requests_finished_by_window_end=1))
    return native, rows, cache


def test_recovered_token_goodput_excludes_late_finished_success_and_keeps_lower_bound(tmp_path):
    native, _, _ = token_cache_fixture(tmp_path)
    fields = mod.recovered_token_columns(native)
    assert fields['recovered_native_window_good_output_tokens_lower_bound'] == 20
    assert fields['recovered_native_goodput_token_s_lower_bound'] == 20/150
    assert 'conservative_lower_bound' in fields['recovered_native_window_goodput_status']
    assert 'not_last_token_timestamp' in fields['recovered_native_window_goodput_status']


@pytest.mark.parametrize('bad', ['sha', 'duplicate_index', 'joint_tokens', 'point_digest'])
def test_recovered_token_cache_binding_and_totals_rejected(tmp_path, bad):
    native, rows, cache = token_cache_fixture(tmp_path)
    if bad == 'sha': native['scalar_requests']['sha256'] = '0'*64
    if bad == 'duplicate_index':
        rows[2]['idx'] = 0; cache.write_text(json.dumps(rows))
        native['scalar_requests']['sha256'] = mod.sha(cache.read_bytes())
    if bad == 'joint_tokens': native['counts']['native_joint_slo_completion_tokens'] = 51
    if bad == 'point_digest': native['bindings']['point_sha256'] = '0'*64
    with pytest.raises(ValueError):
        mod.recovered_token_columns(native)


def auxiliary_fixture(tmp_path, unavailable=False, partial_tail=False):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=mod.sha(path.read_bytes()))

    uuids = ['GPU-'+str(i) for i in range(8)]
    source_dir = tmp_path/'source'
    files = {name+'.py': 'a'*64 for name in ('factory', 'backend', 'power_sampler', 'wrapper')}
    source = write(source_dir/'manifest.json', dict(source_sha256='rev', files=files))
    group = write(tmp_path/'group.json', dict(engine_signature='engine'))
    scope = dict(schema='auxiliary-meter-prospective-scope/v1', created_s=0., used_for_ranking=False,
        selection_independent_of_primary_result=True, all_future_pd_windows=True,
        canonical_columns_modified=False, observer_source=source, script={'path':'observer.py','sha256':'s'*64},
        sampling_interval_s=.1, max_gap_s=1., run_id='run',
        groups=[dict(model_id='model', group=group)])
    scope_ref = write(tmp_path/'scope.json', scope)
    identity = {k:'id' for k in ('image_digest','model_hash','tokenizer_hash','runtime_source_sha256','measurement_source_sha256')}
    point = dict(name='point', system='pdblend', model_id='model', run_id='run', revision='rev',
                 source_manifest=source, engine_identity=dict(identity, fleet_gpu_uuids=uuids))
    directory = tmp_path/'session/windows/point'
    point_ref = write(directory/'point.json', point)
    point_digest = mod.sha(json.dumps(point, sort_keys=True, separators=(',', ':')).encode())
    metrics = dict(service_start_s=10., service_end_s=160., tail_end_s=170.)
    result = write(directory/'result.json', dict(metrics=metrics))
    drain = write(directory/'drain.json', dict(passed=True, tail_end_s=170.))
    receipt_data = dict(point_sha256=point_digest, session_id='session', window_index=0,
        cleanup_passed=not unavailable, artifacts={'point.json':point_ref['sha256']})
    if not unavailable:
        receipt_data.update(result={'metrics':metrics})
        receipt_data['artifacts'].update({'result.json':result['sha256'],'drain.json':drain['sha256']})
    else:
        receipt_data['error'] = 'TimeoutError'
    receipt_ref = write(directory/'receipt.json', receipt_data)
    binding = dict(receipt_path=receipt_ref['path'], receipt_sha256=receipt_ref['sha256'], point_id='point',
                   point_sha256=point_digest, system='pdblend', run_id='run', revision='rev')
    config = dict(auxiliary_only=True, production_factory=True, created_s=1.,
        script=scope['script'], observer_source=dict(manifest=source,source_sha256='rev'),
        execution_source_sha256=['rev'], interval_s=.1, max_gap_s=1., gpu_uuids=uuids,
        session_dir=str(tmp_path/'session'), session_id='session', engine_signature='engine')
    config_ref = write(tmp_path/'aux/config.json', config)
    method = dict(gpu_uuids=uuids, maximum_interpolation_gap_s=1., polling_interval_s=.1, formal_eligible=False)
    method.update({name:dict(path=str(source_dir/(name+'.py')),sha256='a'*64) for name in
                   ('factory','backend','power_sampler','wrapper')})
    method_ref = write(tmp_path/'aux/method.json', method)
    window = dict(receipt=receipt_ref, point='point', available=not unavailable)
    if unavailable:
        window['unavailable_reason'] = 'no_recorded_service_result'
    else:
        window.update(point_artifact=point_ref, result=result, drain=drain, duration_s=150., **metrics,
                      execution_identity=dict(identity, source_sha256='rev', gpu_uuids=uuids))
    stop_ref = write(tmp_path/'aux/stop.json', dict(config=config_ref,
                     no_next_service_until_finalized=True, windows=[dict(window)]))
    window.update(auxiliary_only=True, canonical_modified=False, used_for_ranking=False)
    def phase(start, end):
        duration = end-start
        power = dict(status='complete', coverage_fraction=1., minimum_gpu_coverage_fraction=1.,
            max_gap_s=.1, energy_j=80., per_gpu={u:dict(status='complete', coverage_fraction=1., max_gap_s=.1,
                                                     integral=10.) for u in uuids})
        util = dict(status='complete', coverage_fraction=1., minimum_gpu_coverage_fraction=1.,
            max_gap_s=.1, mean_pct=25., per_gpu={u:dict(status='complete', coverage_fraction=1., max_gap_s=.1,
                                                     integral=25.*duration) for u in uuids})
        return dict(start_s=start,end_s=end,duration_s=duration,power=power,utilization=util)
    if not unavailable:
        window['summary'] = dict(gpu_uuids=uuids,gpu_count=8,gpu_uuid_binding_verified=True,
            maximum_interpolation_gap_s=1.,polling_interval_s=.1,
            utilization_timestamp_semantics='per_device_acquisition_time', power_source_verified=True,
            power_error_affects_window=False,service=phase(10.,160.),tail=phase(160.,170.))
        if partial_tail:
            p=window['summary']['tail']['power'];p.update(status='partial', coverage_fraction=.5,
                minimum_gpu_coverage_fraction=.5,max_gap_s=2.,energy_j=None)
            for v in p['per_gpu'].values():v.update(status='partial',coverage_fraction=.5,max_gap_s=2.,integral=None)
    final = dict(schema='auxiliary-resident-meter/v1',status='completed',auxiliary_only=True,
        canonical_modified=False,restart_count=0,fragment_joining=False,config=config_ref,method=method_ref,
        stop_request=stop_ref,windows=[window],raw_snapshot=dict(path=str(tmp_path/'aux/never-read-raw.json'),sha256='r'*64))
    final_ref = write(tmp_path/'aux/final.json', final)
    artifact = write(tmp_path/'aux/window.json',dict(schema='auxiliary-window-bound-summary/v1',binding=binding,
        scope=scope_ref,final=final_ref,session_id='session',window_index=0))
    annotation = dict(binding=binding,columns=mod.auxiliary_columns(window,config,method_ref,artifact))
    registry = dict(schema=mod.AUXILIARY_SCHEMA,scope=scope_ref,finals=[final_ref],annotations=[annotation],
                    default_columns=mod.annotation_defaults(mod.AUXILIARY_SCHEMA))
    rows=[dict(binding,session_id='session',window_index='0',status='failed' if unavailable else 'measured',
               energy_service_j='' if unavailable else '00099.00100',slo_pass='' if unavailable else 'False'),
          dict(binding,receipt_sha256='history',point_id='history',session_id='old',window_index='2',
               status='measured',energy_service_j='001.2300',slo_pass='False')]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode(), registry, rows


@pytest.mark.parametrize('unavailable,partial_tail',[(False,False),(True,False),(False,True)])
def test_auxiliary_keeps_primary_cells_and_independent_phase_validity(tmp_path, unavailable, partial_tail):
    original,registry,rows=auxiliary_fixture(tmp_path,unavailable,partial_tail)
    result,report=mod.prepare_csv(original,registry)
    output=list(csv.DictReader(io.StringIO(result.decode())))
    assert [{k:r[k] for k in rows[0]} for r in output] == rows
    assert output[1]['auxiliary_status']=='not_scheduled'
    assert output[1]['auxiliary_energy_service_j']==''
    assert output[0]['auxiliary_used_for_ranking']=='False'
    assert output[0]['auxiliary_status']==('unavailable' if unavailable else 'partial' if partial_tail else 'complete')
    assert output[0]['auxiliary_energy_service_j']==('' if unavailable else '80.0')
    assert output[0]['auxiliary_energy_tail_j']==('' if unavailable or partial_tail else '80.0')
    assert mod.prepare_csv(result,registry)[0]==result
    # There is deliberately no raw snapshot file: validation must not reopen it.
    assert not (tmp_path/'aux/never-read-raw.json').exists()


@pytest.mark.parametrize('bad',['drop_window','canonical_interval','artifact_sha','ranking','gpu_identity','csv_session'])
def test_auxiliary_rejects_cherry_pick_identity_and_interval_changes(tmp_path,bad):
    original,registry,rows=auxiliary_fixture(tmp_path)
    if bad=='drop_window': registry['annotations']=[]
    if bad=='artifact_sha': registry['annotations'][0]['columns']['auxiliary_artifact_sha256']='0'*64
    if bad=='ranking': registry['annotations'][0]['columns']['auxiliary_used_for_ranking']=True
    if bad=='csv_session': original=original.replace(b',session,0,',b',wrong-session,0,')
    if bad in ('canonical_interval','gpu_identity'):
        final_ref=registry['finals'][0];path=Path(final_ref['path']);final=json.loads(path.read_text())
        if bad=='canonical_interval': final['windows'][0]['summary']['service']['start_s']=11.
        if bad=='gpu_identity': final['windows'][0]['summary']['gpu_uuids'].reverse()
        path.write_text(json.dumps(final));final_ref['sha256']=mod.sha(path.read_bytes())
        cols=registry['annotations'][0]['columns'];ap=Path(cols['auxiliary_artifact_path']);artifact=json.loads(ap.read_text())
        artifact['final']=final_ref;ap.write_text(json.dumps(artifact));cols['auxiliary_artifact_sha256']=mod.sha(ap.read_bytes())
    with pytest.raises(ValueError): mod.prepare_csv(original,registry)


@pytest.mark.parametrize('bad', ['native_sha', 'count', 'sample_scope', 'coverage', 'tail', 'ranking', 'method', 'origin', 'slo'])
def test_recovery_rejects_false_binding_coverage_or_completion(tmp_path, bad):
    def corrupt(native, power):
        if bad=='native_sha': native['native_completion']['sha256']='bad'
        if bad=='count': native['counts']['native_successful_requests']=3
        if bad=='sample_scope': native['native_ttft']['samples']=4
        if bad=='coverage': power['service']['power']['coverage_fraction']=.99
        if bad=='tail': power['outer_drain_complete']=True
        if bad=='ranking': power['used_for_ranking']=True
        if bad=='method': power['original_method']['sha256']='wrong'
        if bad=='origin': power['service_start_s']=2.
        if bad=='slo': native['native_request_slo_pass']=True
    f=recovered_fixture(tmp_path,corrupt); original=f[0].read_bytes()
    with pytest.raises(ValueError): execute(f,apply=True)
    assert f[0].read_bytes()==original


def energy_fixture(tmp_path):
    f = fixture(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    meter = run / "comparison-metering.json"
    canonical = dict(covered_energy_j=320.0, energy_j=None, coverage_fraction=0.4 / 1.5, max_gap_s=1.1)
    meter.write_text(json.dumps({"service": {"power": canonical, "start_s": 0.5, "end_s": 2.0},
                                "tail": {"power": canonical, "start_s": 2.0, "end_s": 3.5}}))
    journal_sha = "a" * 64
    binding = update_receipt(f, {"point_sha256": "point-digest", "artifacts": {
        "run/power.samples.jsonl.gz": journal_sha,
        "run/comparison-metering.json": mod.sha(meter.read_bytes()),
    }})
    projection = tmp_path / "projection.json"
    timestamps = [0.0, 0.5, 1.6, 2.0, 3.1, 3.5, 4.0]
    projection.write_text(json.dumps({"source_sha256": journal_sha, "selected": {
        "samples": [[t, [100.0] * 8] for t in timestamps],
        "power_metadata": [{"read_finished_s": [t] * 8} for t in timestamps]}}))
    phases = {}
    for name, start, end in (("service", 0.5, 2.0), ("tail", 2.0, 3.5)):
        phases[name] = dict(canonical_energy_j=None, all_gpu_boundaries_bracketed=True,
            strict_covered_energy_j=320.0, strict_coverage_fraction=0.4 / 1.5, max_gap_s=1.1,
            start_s=start, end_s=end, trapezoidal_estimate_j=1200.0, estimated_gap_contribution_j=880.0,
            per_gpu=[dict(gpu=i, gap_interpolated_j=110.0, gaps=[dict(read_before_s=start,
                read_after_s=start + 1.1, left_w=100.0, right_w=100.0)]) for i in range(8)])
    diagnostic = tmp_path / "energy.json"
    diagnostic.write_text(json.dumps(dict(schema="pdblend-sampling-gap-energy-analysis/v1",
        used_for_ranking=False, receipt={"path": binding["receipt_path"], "sha256": binding["receipt_sha256"]},
        metering={"path": str(meter), "sha256": mod.sha(meter.read_bytes())},
        projection={"path": str(projection), "sha256": mod.sha(projection.read_bytes())},
        source_journal={"path": str(run / "power.samples.jsonl.gz"), "sha256": journal_sha}, phases=phases)))
    columns = dict(zip(mod.ENERGY_COLUMNS, (
        "trapezoidal_estimate_over_sampling_gaps", 1200.0, 1200.0, 2400.0, 880.0, 880.0,
        0.4 / 1.5, 0.4 / 1.5, 1.1, 1.1, True,
        "per_gpu_acquisition_time_linear_trapezoid_including_gaps_over_1s", False,
        str(diagnostic), mod.sha(diagnostic.read_bytes()),
    )))
    f[4].update(schema=mod.ENERGY_SCHEMA,
                annotations=[dict(binding=binding, columns=columns)],
                default_columns={k: "not_annotated" if k == mod.ENERGY_COLUMNS[0] else "" for k in mod.ENERGY_COLUMNS})
    f[1].write_text(json.dumps(f[4]))
    return f


def test_gap_estimates_are_separate_and_do_not_replace_energy_or_slo(tmp_path):
    f = energy_fixture(tmp_path)
    before = list(csv.DictReader(io.StringIO(f[0].read_text(), newline="")))
    execute(f, apply=True)
    rows = list(csv.DictReader(io.StringIO(f[0].read_text(), newline="")))
    assert [{k: r[k] for k in before[0]} for r in rows] == before
    assert rows[0]["analysis_energy_service_trapezoid_j"] == "1200.0"
    assert rows[0]["analysis_energy_estimate_eligible_for_ranking"] == "False"
    assert execute(f, apply=True)["changed"] is False


@pytest.mark.parametrize("key,value", [
    ("analysis_energy_estimate_eligible_for_ranking", True),
    ("analysis_energy_service_trapezoid_j", 1201.0),
    ("analysis_energy_estimate_boundaries_bracketed", False),
])
def test_estimate_ranking_or_value_changes_rejected(tmp_path, key, value):
    f = energy_fixture(tmp_path)
    before = f[0].read_bytes()
    f[4]["annotations"][0]["columns"][key] = value
    f[1].write_text(json.dumps(f[4]))
    with pytest.raises(ValueError, match="verified estimate"):
        execute(f, apply=True)
    assert f[0].read_bytes() == before


def test_unbracketed_energy_projection_cannot_be_extrapolated(tmp_path):
    f = energy_fixture(tmp_path)
    before = f[0].read_bytes()
    columns = f[4]["annotations"][0]["columns"]
    diagnostic_path = Path(columns[mod.ENERGY_COLUMNS[-2]])
    diagnostic = json.loads(diagnostic_path.read_text())
    projection_path = Path(diagnostic["projection"]["path"])
    projection = json.loads(projection_path.read_text())
    for key in ("samples", "power_metadata"):
        projection["selected"][key] = projection["selected"][key][2:]
    projection_path.write_text(json.dumps(projection))
    diagnostic["projection"]["sha256"] = mod.sha(projection_path.read_bytes())
    diagnostic_path.write_text(json.dumps(diagnostic))
    columns[mod.ENERGY_COLUMNS[-1]] = mod.sha(diagnostic_path.read_bytes())
    f[1].write_text(json.dumps(f[4]))
    with pytest.raises(ValueError, match="does not bracket"):
        execute(f, apply=True)
    assert f[0].read_bytes() == before


def io_fixture(tmp_path):
    f = fixture(tmp_path)
    binding = update_receipt(f, {"point_sha256": "point-digest", "result": {
        "metrics": {"service_start_s": 100.0, "service_end_s": 250.0}}})
    diagnostic = tmp_path / "io.json"
    diagnostic.write_text(json.dumps(dict(schema="independent-review-io-incident/v1",
        mixed_window=dict(receipt={"path": binding["receipt_path"], "sha256": binding["receipt_sha256"]},
            service_start_s=100.0, service_end_s=250.0, overlap="unknown_possible", causal_effect="not_determined"),
        operation=dict(wall_start_s=None, wall_end_s=None, reported_tool_elapsed_s=2.16172),
        files=[dict(size_bytes=105284792), dict(size_bytes=150192574)], total_size_bytes=255477366)))
    columns = dict(zip(mod.IO_COLUMNS, (
        "possible_overlap_timing_unknown", 255477366, 2.16172, "", "unknown",
        "reported_command_elapsed_not_measured_io_busy_time", str(diagnostic), mod.sha(diagnostic.read_bytes()),
    )))
    f[4].update(schema=mod.IO_SCHEMA, annotations=[dict(binding=binding, columns=columns)],
                default_columns={k: "not_annotated" if k == mod.IO_COLUMNS[0] else "" for k in mod.IO_COLUMNS})
    f[1].write_text(json.dumps(f[4]))
    return f


def test_io_unknown_overlap_stays_blank_and_old_cells_preserved(tmp_path):
    f = io_fixture(tmp_path)
    before = list(csv.DictReader(io.StringIO(f[0].read_text(), newline="")))
    execute(f, apply=True)
    rows = list(csv.DictReader(io.StringIO(f[0].read_text(), newline="")))
    assert [{k: r[k] for k in before[0]} for r in rows] == before
    assert rows[0]["concurrent_audit_io_overlap_s"] == ""
    assert rows[0]["concurrent_audit_io_command_elapsed_s"] == "2.16172"
    assert rows[0]["concurrent_audit_io_input_bytes"] == "255477366"
    assert execute(f, apply=True)["changed"] is False


def test_io_elapsed_must_not_be_used_as_overlap(tmp_path):
    f = io_fixture(tmp_path)
    f[4]["annotations"][0]["columns"]["concurrent_audit_io_overlap_s"] = 2.16172
    f[1].write_text(json.dumps(f[4]))
    with pytest.raises(ValueError, match="invents overlap"):
        execute(f, apply=True)
