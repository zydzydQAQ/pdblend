from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.engine.launcher import InstanceSpec
from pdblend.measure.backends import BackendError, physical_gpu
from pdblend.profile.identity import ProfileKey
from pdblend.profile.wave import ProfileWave
from pdblend.profile.acceptance import validate_parallel_interference
from pdblend.profile.parallel import common_window_overlap


def _profiler(tmp_path: Path, member: str, uuids: list[str]):
    out = tmp_path / f"out-{member}"
    out.mkdir()
    key = ProfileKey("pdblend", "Qwen2.5-7B-Instruct", "vllm-test", "test", 1)
    return SimpleNamespace(
        raw={"environment": {"gpu_uuids": uuids}},
        profile_key=key,
        specs=[SimpleNamespace(instance_id=f"{member}-i0", base_url="http://127.0.0.1")],
        out_dir=out,
        _checkpoint=lambda: None,
    )


def _row(start: float, end: float, step: float = 1.0, power: float = 100.0) -> dict:
    return {"step_seconds": step, "power_w": power,
            "repeats": [{"start_s": start, "end_s": end}]}


def _seed_peer(root: Path, *, uuids=("GPU-b",), start=0.0, end=4.0, error=False):
    if error:
        (root / "b.error.json").write_text(json.dumps({"error": "peer failed"}))
        return
    (root / "b.ready.json").write_text(json.dumps({"gpu_uuids": list(uuids)}))
    payload = {"instances": [_row(start, end)], "point": {"frequency": 2100}}
    (root / "b.isolated.json").write_text(json.dumps(payload))
    (root / "b.parallel_ready.json").write_text(json.dumps({"time": 1}))
    (root / "b.parallel.json").write_text(json.dumps(payload))


async def _qualify(tmp_path: Path, *, peer_uuids=("GPU-b",), peer_start=0.0,
                   peer_end=4.0, peer_error=False):
    root = tmp_path / "wave"
    root.mkdir(parents=True)
    (root / "wave.json").write_text(json.dumps({"members": ["a", "b"]}))
    _seed_peer(root, uuids=peer_uuids, start=peer_start, end=peer_end, error=peer_error)
    wave = ProfileWave(root, "a", timeout_s=0.2)
    profiler = _profiler(tmp_path, "a", ["GPU-a"])

    async def fake_probe(_profiler, phase):
        return {"instances": [_row(0.0, 4.0)],
                "point": {"frequency": 2100, "batch": 8, "context": 1024},
                "gpu_uuids": ["GPU-a"]}

    wave.probe = fake_probe
    await wave.qualify(profiler)
    return wave, profiler


def test_wave_cross_member_measured_comparison_and_serialization_fallback(tmp_path):
    wave, profiler = asyncio.run(_qualify(tmp_path))
    evidence = json.loads((profiler.out_dir / "samples/external-interference.json").read_text())
    assert evidence["passed"] is True
    assert evidence["overlapping_windows"] is True
    assert evidence["comparisons"][0]["passed"] is True

    serial_wave, serial_profiler = asyncio.run(
        _qualify(tmp_path / "serial", peer_start=8.0, peer_end=9.0))
    serial = json.loads((serial_profiler.out_dir / "samples/external-interference.json").read_text())
    assert serial_wave.parallel is False
    assert serial["fallback"] == "serial_cohort"
    assert serial["overlapping_windows"] is False


def test_wave_rejects_peer_failure_and_uuid_overlap(tmp_path):
    async def peer_failure():
        await _qualify(tmp_path / "failure", peer_error=True)

    with pytest.raises(RuntimeError, match="peer failed"):
        asyncio.run(peer_failure())

    async def overlap():
        await _qualify(tmp_path / "overlap", peer_uuids=("GPU-a",))

    with pytest.raises(RuntimeError, match="overlaps physical GPUs"):
        asyncio.run(overlap())


def test_wave_receipt_validates_nested_member_instances_end_to_end(tmp_path):
    root = tmp_path / "wave"
    root.mkdir(parents=True)
    (root / "wave.json").write_text(json.dumps({
        "members": ["a", "b"], "cohort_id": "cohort-test", "coordinator": True}))
    _seed_peer(root)
    wave = ProfileWave(root, "a", timeout_s=0.2)
    profiler = _profiler(tmp_path, "a", ["GPU-a"])

    async def fake_probe(_profiler, phase):
        return {"instances": [_row(0.0, 4.0)],
                "point": {"frequency": 2100, "batch": 8, "context": 1024},
                "gpu_uuids": ["GPU-a"]}

    wave.probe = fake_probe
    asyncio.run(wave.qualify_external(profiler))
    result = validate_parallel_interference(profiler.raw, profiler.out_dir)
    assert result["passed"] and result["formal_eligible"]


def test_physical_gpu_uses_uuid_lease_and_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("PDBLEND_GPU_UUIDS", "GPU-a,GPU-b")
    assert physical_gpu(0) == "GPU-a"
    assert physical_gpu(1) == "GPU-b"
    with pytest.raises(BackendError):
        physical_gpu(2)
    with pytest.raises(BackendError):
        physical_gpu(-1)


def test_launcher_environment_uses_physical_uuid_mapping(monkeypatch):
    monkeypatch.setenv("PDBLEND_GPU_UUIDS", "GPU-a,GPU-b")
    spec = InstanceSpec("i0", (1, 0), 8100, "/models/Qwen", tp=2, kv_connector=None)
    env = spec.environment()
    # The launcher runs inside the container and receives local ordinals;
    # physical_gpu() maps those ordinals to leased UUIDs for NVML operations.
    assert env["CUDA_VISIBLE_DEVICES"] == "1,0"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_pairwise_overlap_does_not_qualify_whole_cohort():
    result = common_window_overlap([_row(0, 5), _row(-3, 2), _row(3, 8)])
    assert result['passed'] is False
    assert result['overlap_seconds'] == [0.]
    assert common_window_overlap([_row(0, 5), _row(1, 6), _row(2, 7)])['passed']


def test_all_repeats_require_real_common_windows():
    first = _row(0, 5)
    first['repeats'].append({'start_s': 10, 'end_s': 15})
    second = _row(1, 6)
    second['repeats'].append({'start_s': 14, 'end_s': 19})
    result = common_window_overlap([first, second])
    assert result['overlap_seconds'] == [4, 1]
    assert not result['passed']
    assert not common_window_overlap([first])['passed']
    assert not common_window_overlap([first, _row(0, 5)])['passed']
    assert not common_window_overlap([_row(float('nan'), 5), _row(0, 5)])['passed']


def test_parallel_probe_waits_for_all_local_and_remote_settled_instances(tmp_path, monkeypatch):
    import time
    import pdblend.profile.wave as module

    class Client:
        def __init__(self, iid, _url): self.iid = iid
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass

    monkeypatch.setattr(module, 'EngineClient', Client)
    root = tmp_path/'synchronized'; root.mkdir()
    (root/'wave.json').write_text(json.dumps(dict(members=['a', 'b'], synchronize_parallel_windows=True)))
    settled, observed = {}, []

    async def scenario():
        async def fake_batch(client, *_args, before_measure=None):
            result = dict(step_seconds=1., power_w=100., repeats=[])
            for repeat in range(3):
                await asyncio.sleep(.02 if client.iid == 'a-i0' else .07)
                settled.setdefault(repeat, set()).add(client.iid)
                await before_measure(repeat)
                observed.append((repeat, set(settled[repeat])))
                now = time.time()
                result['repeats'].append(dict(start_s=now, end_s=now+5.))
            return result

        def profiler(member, count):
            return SimpleNamespace(specs=[SimpleNamespace(instance_id=f'{member}-i{i}',
                base_url='unused', gpus=[i]) for i in range(count)],
                raw={'environment': {'gpu_uuids': [member]}}, _lock=lambda *_args: None,
                _decode_batch=fake_batch)

        return await asyncio.gather(ProfileWave(root, 'a', timeout_s=3).probe(profiler('a', 2), 'parallel'),
                                    ProfileWave(root, 'b', timeout_s=3).probe(profiler('b', 1), 'parallel'))

    results = asyncio.run(scenario())
    assert len(observed) == 9
    assert all(ready == {'a-i0', 'a-i1', 'b-i0'} for _, ready in observed)
    assert common_window_overlap([inst for member in results for inst in member['instances']])['passed']


def test_verifier_recomputes_common_windows_instead_of_trusting_flag(tmp_path):
    from hashlib import sha256
    wave, profiler = asyncio.run(_qualify(tmp_path))
    path = profiler.out_dir/'samples/external-interference.json'
    evidence = json.loads(path.read_text())
    evidence.update(cohort_id='test', cross_job=True, members=['a', 'b', 'c'], overlapping_windows=True)
    evidence['parallel'] = [{'instances': [_row(0, 5)]}, {'instances': [_row(-3, 2)]},
                            {'instances': [_row(3, 8)]}]
    evidence['isolated'] = evidence['parallel']
    path.write_text(json.dumps(evidence))
    profiler.raw['external_interference'].update(cross_job=True, cohort_id='test',
        samples_sha256=sha256(path.read_bytes()).hexdigest())
    result = validate_parallel_interference(profiler.raw, profiler.out_dir)
    assert not result['passed']
    assert any('common windows' in failure for failure in result['failures'])


@pytest.mark.parametrize('seconds', [4.9, 0, True, float('nan'), float('inf')])
def test_qualification_duration_cannot_weaken_measurement_minimum(tmp_path, seconds):
    (tmp_path/'wave.json').write_text(json.dumps(dict(members=['a'], qualification_measure_s=seconds)))
    with pytest.raises(ValueError, match='at least 5 seconds'):
        ProfileWave(tmp_path, 'a')


def test_longer_probe_keeps_ordinary_measurement_duration(tmp_path, monkeypatch):
    import pdblend.profile.wave as module

    class Client:
        def __init__(self, *_args): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): pass

    class Profiler:
        decode_measure_s = 5.
        specs = [SimpleNamespace(instance_id='a', base_url='unused', gpus=[0])]
        raw = dict(environment=dict(gpu_uuids=['GPU-a']))
        def _lock(self, *_args): pass
        async def _decode_batch(self, *_args, **_kwargs):
            return _row(0, self.decode_measure_s)

    monkeypatch.setattr(module, 'EngineClient', Client)
    (tmp_path/'wave.json').write_text(json.dumps(dict(members=['a'], qualification_measure_s=8.)))
    profiler = Profiler()
    wave = ProfileWave(tmp_path, 'a')
    measured = asyncio.run(wave.probe(profiler, 'isolated'))
    assert measured['instances'][0]['repeats'][0]['end_s'] == 8.
    assert profiler.decode_measure_s == 5.
