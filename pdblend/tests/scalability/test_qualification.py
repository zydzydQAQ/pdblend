import asyncio
import copy
from contextlib import nullcontext
import json

import pytest

from ecopadg.scalability import qualification as q
from ecopadg.serving.interconnect import InterconnectTopology


def test_directional_pair_selection_covers_real_topology_classes():
    topology = InterconnectTopology.parse("GPU0 X PIX SYS\nGPU1 PIX X SYS\nGPU2 SYS SYS X\n")
    instances = [dict(id=str(g), gpus=[g]) for g in range(3)]
    pairs = q.representative_pairs(instances, topology)
    assert {(r["link_class"], r["direction"]) for r in pairs} == {
        ("PIX", "ascending"), ("PIX", "descending"), ("SYS", "ascending"), ("SYS", "descending")}
    assert all(r["source"]["id"] != r["target"]["id"] for r in pairs)


def test_active_frequency_cannot_be_certified_from_commands_or_idle_clocks():
    def samples(freq=1500, active=True, idle=False):
        return [dict(observed_mhz=freq, engine_active=active, clock_idle=idle, elapsed_s=t) for t in (.4, .5)]
    assert q.active_frequency_pass(samples(), 1500)
    assert not q.active_frequency_pass(samples(active=False), 1500)
    assert not q.active_frequency_pass(samples(idle=True), 1500)
    assert not q.active_frequency_pass(samples(freq=900), 1500)
    assert not q.active_frequency_pass(samples()[:1], 1500)
    assert not q.active_frequency_pass([dict(x, elapsed_s=.1) for x in samples()], 1500)


def test_clock_restore_never_guesses_original_lock_from_idle_frequency():
    with pytest.raises(ValueError, match="explicitly cover"):
        q.clock_restore_targets({}, [0, 1, 2])
    targets = {str(g): None for g in range(3)}
    with pytest.raises(ValueError, match="evidence basis"):
        q.clock_restore_targets(dict(qualification_clock_restore=targets), [0, 1, 2])
    assert q.clock_restore_targets(dict(qualification_clock_restore=targets,
                                        qualification_clock_baseline="reset_before_qualification"), [0, 1, 2]) == (targets, True)
    with pytest.raises(ValueError, match="unlocked"):
        q.clock_restore_targets(dict(qualification_clock_restore={**targets, "0": 1500},
                                      qualification_clock_baseline="reset_before_qualification"), [0, 1, 2])


def test_output_comparison_requires_exact_real_token_work():
    output = dict(token_ids=[4, 8], usage=dict(completion_tokens=2))
    assert q.output_matches(output, [4, 8], 2)
    assert not q.output_matches(output, [4, 9], 2)
    assert not q.output_matches(dict(output, usage=dict(completion_tokens=1)), [4, 8], 2)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """CPU-only protocol simulation: these objects never reach a GPU or socket."""
    class Rig:
        profile_errors = []
        fail_completion = False
        reset_calls = []
        sampler_gpu_ids = None
        client_instances = None
    env = Rig()
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps(dict(engine_image="image-test")))
    topology = tmp_path / "topology.txt"
    topology.write_text("GPU0 X PIX SYS SYS\nGPU1 PIX X SYS SYS\nGPU2 SYS SYS X PIX\nGPU3 SYS SYS PIX X\n")
    env.config = dict(profiles=str(profiles), interconnect=str(topology),
        instances=[dict(id=f"e{g}", gpus=[g], tp=1, url=f"http://invalid:{30000+g}") for g in range(4)],
        qualification_clock_restore={str(g): None for g in range(4)},
        qualification_clock_baseline="reset_before_qualification")
    env.states = {i["id"]: dict(role="mixed", mode="continuous", admit_prefill=True, admit_decode=True,
        generation=0, acknowledged_generation=0, accepting=True, active=0, running=0, waiting=0,
        kv_allocations={}, transfer_allocations={}, transfer_buffered_tensors=0, transport_healthy=True)
        for i in env.config["instances"]}
    env.original = copy.deepcopy(env.states)

    async def inspect(config):
        return [dict(instance_id=i["id"], errors=[], tp=1, dtype="bfloat16", model="Qwen2.5-14B-Instruct",
                     image_id="image-test", cuda_visible_devices=str(i["gpus"][0]),
                     source_files_at_import={"source": "immutable"}, runtime=copy.deepcopy(env.states[i["id"]]))
                for i in config["instances"]]

    class Hardware:
        def __init__(self, power_mode):
            assert power_mode == "instant"
        def reset_clock(self, gpu):
            env.reset_calls.append(gpu)
        def set_clock(self, gpu, frequency):
            pass

    class Clocks:
        def __init__(self, backend, gpus):
            self.backend, self.gpus = backend, list(gpus)
            self.pool = None
            self.lock = asyncio.Lock()
        async def set(self, gpus, frequency, verify_rise=False):
            assert set(gpus) <= set(self.gpus)
        async def close(self):
            for gpu in self.gpus:
                self.backend.reset_clock(gpu)

    class Sampler:
        def __init__(self, gpus, **kwargs):
            env.sampler_gpu_ids = list(gpus)
            self.samples = [(1., [1.] * 8), (2., [1.] * 8)]
            self.error = None
            self.power_source = dict(mode="instant", field_id=186)
            self.power_metadata = []
            self.frequency_samples = []
            self.utilization_samples = []
        def start(self):
            pass
        def stop(self):
            pass

    class Client:
        def __init__(self, session, raw):
            self.owned = {}
            self.raw = raw
        async def call(self, instance, path, body=None, rid=None):
            state = env.states[instance["id"]]
            self.raw["rpc"].append(dict(instance_id=instance["id"], path=path))
            if path == "/runtime":
                return copy.deepcopy(state)
            if path == "/prepare-peers":
                return dict(ready=True)
            if path == "/v1/completions":
                if env.fail_completion:
                    raise RuntimeError("injected engine failure")
                if rid and rid.startswith("pdb:"):
                    _, nonce, phase, source, target = rid.split(":")
                    dest = env.states[target]
                    if phase == "p":
                        dest["transfer_buffered_tensors"] = 1
                        dest["transfer_allocations"] = {nonce: 64}
                    else:
                        dest["transfer_buffered_tensors"] = 0
                        dest["transfer_allocations"] = {}
                self.owned.pop(rid, None)
                return dict(token_ids=list(range(body["max_tokens"])), usage=dict(completion_tokens=body["max_tokens"]))
            if path == "/cancel":
                state["transfer_buffered_tensors"] = 0
                state["transfer_allocations"] = {}
                return dict(transfers=[dict(buffered_tensors=0, inflight_receives=0, inflight_sends=0, listener_alive=True)])
            raise AssertionError(path)
        async def control(self, instance, **changes):
            state = env.states[instance["id"]]
            state.update(changes, accepting=True, generation=state["generation"] + 1,
                         acknowledged_generation=state["generation"] + 1)
            return copy.deepcopy(state)
        async def drained(self, instance, native=False):
            state = env.states[instance["id"]]
            assert q.quiescent(state)
            if native:
                state.update(accepting=False, admit_prefill=False, generation=state["generation"] + 1,
                             acknowledged_generation=state["generation"] + 1)
            return copy.deepcopy(state)
        async def cleanup_owned(self):
            for rid, instance in list(self.owned.items()):
                await self.call(instance, "/cancel", dict(request_id=rid))
                self.owned.pop(rid, None)

    async def frequency(client, instance, hardware, clocks, frequency):
        return dict(instance_id=instance["id"], requested_mhz=frequency, passed=True,
                    output=await client.call(instance, "/v1/completions", q.BODY))

    monkeypatch.setattr(q, "inspect_engines", inspect)
    monkeypatch.setattr(q, "check_profile_inputs", lambda config: list(env.profile_errors))
    monkeypatch.setattr(q, 'measured_source_errors', lambda config,records: list(getattr(env,'source_errors',[])))
    async def topology():
        return 'fixture topology'
    monkeypatch.setattr(q,'read_live_topology',topology)
    monkeypatch.setattr(q,'topology_errors',lambda config,text: [])
    monkeypatch.setattr(q, "PynvmlBackend", Hardware)
    monkeypatch.setattr(q, "ClockOwner", Clocks)
    monkeypatch.setattr(q, "PowerSampler", Sampler)
    monkeypatch.setattr(q, "QualificationClient", Client)
    monkeypatch.setattr(q, "frequency_probe", frequency)
    monkeypatch.setattr(q, "node_lease", nullcontext)
    env.out = tmp_path / "out"
    return env


def test_missing_profile_artifacts_block_formal_but_keep_real_mechanism_results(rig):
    rig.profile_errors = ["missing evidence: historical-raw.json"]
    proof = asyncio.run(q.qualify(rig.config, rig.out))
    assert proof["passed"] is False and proof["checks"]["profile_coverage"] is False
    assert all(proof["checks"][key] for key in q.REQUIRED_CHECKS if key != "profile_coverage")
    raw = json.loads((rig.out / "raw.json").read_text())
    assert len(raw["frequency"]) == 16 and raw["cleanup_complete"] is True
    assert all(row["held_import"]["transfer_buffered_tensors"] == 1 for row in raw["pd"])


def test_old_source_cannot_be_qualified_by_fresh_short_probes(rig):
    rig.source_errors=['profile engine-source mismatch']
    proof=asyncio.run(q.qualify(rig.config,rig.out))
    assert not proof['passed'] and not proof['checks']['profile_coverage']
    assert proof['checks']['mixed_output'] and proof['checks']['pd_output']
    assert 'profile engine-source mismatch' in proof['profile_errors']
    raw=json.loads((rig.out/'raw.json').read_text())
    assert len(raw["setup_clock_resets"]) == 4
    assert rig.reset_calls == [0, 1, 2, 3] * 2
    assert rig.sampler_gpu_ids == list(range(8))
    assert proof["artifacts"][str((rig.out / "raw.json").resolve())] == q.sha256(rig.out / "raw.json")


def test_smoke_never_promotes_to_full_hardware_qualification(rig):
    proof = asyncio.run(q.qualify(rig.config, rig.out, smoke=True))
    raw = json.loads((rig.out / "raw.json").read_text())
    assert proof["passed"] is False and proof["smoke"] is True
    assert len(raw["mixed"]) == 3 and len(raw["frequency"]) == 3
    assert raw["allocated_gpu_ids"] == [0, 1, 2]
    assert 3 not in rig.reset_calls


def test_engine_exception_preserves_raw_power_and_restores_original_controls(rig):
    rig.fail_completion = True
    proof = asyncio.run(q.qualify(rig.config, rig.out))
    raw = json.loads((rig.out / "raw.json").read_text())
    assert proof["passed"] is False
    assert "injected engine failure" in " ".join(proof["errors"])
    assert raw["power_samples"] and raw["cleanup_complete"] is True
    for instance_id, state in rig.states.items():
        assert all(state[k] == rig.original[instance_id][k] for k in q.CONTROL_FIELDS)
    assert rig.reset_calls == [0, 1, 2, 3] * 2


def test_complete_qualification_requires_every_check_and_cannot_overwrite(rig):
    proof = asyncio.run(q.qualify(rig.config, rig.out))
    assert proof["passed"] is True and set(proof["checks"]) == set(q.REQUIRED_CHECKS)
    assert all(proof["checks"].values()) and proof["cleanup_complete"] is True
    with pytest.raises(FileExistsError):
        asyncio.run(q.qualify(rig.config, rig.out))
