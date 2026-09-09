import copy
import fcntl
import os
from pathlib import Path

import pytest

import dispatcher as d


def evidence(label="C", observed=0):
    identity = dict(device=7, inode=42)
    return dict(schema=d.SCHEMA, hostname=d.HOSTS[label]["hostname"], observed_s=observed,
                eligible_snapshot=True, errors=[], blocker_count=0, blockers=[],
                node_lease=dict(free=True, after=dict(identity=identity)),
                gpu=[dict(index=i, uuid=f"GPU-{i}", utilization_percent=0) for i in range(8)])


@pytest.mark.parametrize("argv", [
    ["python3", "-u", "/root/workspace/pdblend-next-v1/campaign/A/continue_p6_load_v3.py"],
    ["python3", "campaign/foo/waiting_continue.py"],
    ["python3", "campaign/foo/baseline_qualification.py"],
    ["python3", "campaign/foo/run.py", "--out", "reports/audit"],
    ["bash", "-c", "sleep 10; python3 campaign/foo/qualification.py"],
    ["python3", "campaign/foo/audit_and_launch.py"],
    ["python3", "campaign/foo/audit.py", "--run"],
])
def test_waiting_and_qualification_owners_are_never_idle(argv):
    assert d.classify_process(argv) == "gpu_work"


@pytest.mark.parametrize("argv,kind", [
    (["python3", "-m", "ecopadg.serving.engine", "--config", "campaign/qualification/engine.json"], "resident_engine"),
    (["python3", "/root/workspace/pdblend-next-v1/campaign/B32B-baseline-fixed-window-preparation-v1/legacy-observation-candidate/engine.py", "--config", "/runtime/engine.json"], "resident_engine"),
    (["python3", "-B", "campaign/C/mirror_progress_p4_until_complete_v2.py"], "read_only"),
    (["python3", "campaign/B/collect_results.py"], "read_only"),
    (["python3", "campaign/A/report.py", "--source", "campaign/runner/status.json"], "read_only"),
    (["python3", "campaign/A/audit.py"], "read_only"),
])
def test_resident_engines_and_read_only_observers_are_separate(argv, kind):
    assert d.classify_process(argv) == kind


def test_three_distinct_spaced_zero_observations_required():
    state = d.new_state()
    original = copy.deepcopy(state)
    state = d.observe(state, "C", evidence(observed=0), received_s=0)
    assert original == d.new_state() and state["candidate"] is None
    state = d.observe(state, "C", evidence(observed=1), received_s=1)
    assert state["hosts"]["C"]["count"] == 1
    state = d.observe(state, "C", evidence(observed=10), received_s=10)
    assert state["candidate"] is None
    state = d.observe(state, "C", evidence(observed=20), received_s=20)
    assert state["candidate"] == "C"
    assert state["recommendation_only"] is True


@pytest.mark.parametrize("change", ["busy_gpu", "lease", "qualification", "sensor", "wrong_host"])
def test_intervening_activity_or_missing_evidence_resets_stability(change):
    state = d.observe(d.new_state(), "C", evidence(observed=0), received_s=0)
    state = d.observe(state, "C", evidence(observed=10), received_s=10)
    bad = evidence(observed=20)
    if change == "busy_gpu":
        bad["gpu"][7]["utilization_percent"] = 1
    elif change == "lease":
        bad["node_lease"]["free"] = False
    elif change == "qualification":
        bad["blocker_count"] = 1
    elif change == "sensor":
        bad["errors"] = ["nvidia-smi failed"]
    else:
        bad["hostname"] = d.HOSTS["B"]["hostname"]
    state = d.observe(state, "C", bad, received_s=20)
    assert state["hosts"]["C"]["count"] == 0 and state["candidate"] is None
    state = d.observe(state, "C", evidence(observed=30), received_s=30)
    assert state["hosts"]["C"]["count"] == 1


def test_duplicate_snapshot_and_changed_gpu_identity_do_not_certify():
    state = d.observe(d.new_state(), "C", evidence(observed=0), received_s=0)
    state = d.observe(state, "C", evidence(observed=0), received_s=10)
    assert state["hosts"]["C"]["count"] == 0
    state = d.observe(state, "C", evidence(observed=20), received_s=20)
    changed = evidence(observed=30)
    changed["gpu"][0]["uuid"] = "GPU-replaced"
    state = d.observe(state, "C", changed, received_s=30)
    assert state["hosts"]["C"]["count"] == 1


def test_kernel_lock_parsing_includes_waiters_and_does_not_confuse_inode():
    device = os.makedev(259, 3)
    text = "1: FLOCK ADVISORY WRITE 99 103:03:42 0 EOF\n2: -> FLOCK ADVISORY WRITE 100 103:03:42 0 EOF\n3: FLOCK ADVISORY WRITE 101 103:03:43 0 EOF\n"
    locks = d.parse_locks(text, device, 42)
    assert {value["pid"] for value in locks} == {99, 100}
    assert any(value["waiting"] for value in locks)


def test_probe_never_creates_missing_lock(tmp_path):
    path = tmp_path / "not-created.lock"
    result = d.node_lease_probe(path, tmp_path)
    assert result["free"] is True and not path.exists()


def test_gpu_sensor_requires_all_eight_unique_boards_and_known_values():
    valid = "\n".join(f"{i}, GPU-{i}, 0" for i in range(8))
    assert len(d.parse_gpu_csv(valid)) == 8
    for bad in ("\n".join(valid.splitlines()[:-1]), valid.replace("GPU-1", "GPU-0"),
                valid.replace("7, GPU-7, 0", "7, GPU-7, N/A")):
        with pytest.raises(ValueError):
            d.parse_gpu_csv(bad)


def test_process_inventory_ignores_zombies_and_tracks_waiting_owner(tmp_path):
    def process(pid, argv, state="S"):
        directory = tmp_path / str(pid)
        directory.mkdir()
        fields = [state, "1"] + ["0"] * 17 + ["123"] + ["0"] * 8
        (directory / "stat").write_text(f"{pid} (test process) " + " ".join(fields))
        (directory / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
    process(1, ["python3", "campaign/waiting_continue.py"])
    process(2, ["python3", "campaign/qualification.py"], state="Z")
    process(3, ["python3", "-m", "ecopadg.serving.engine"])
    process(4, ["python3", "campaign/mirror_status.py"])
    result = d.process_inventory(tmp_path, own_pid=100)
    assert [value["pid"] for value in result["blockers"]] == [1]
    assert result["resident_engine_count"] == result["read_only_count"] == 1
    assert not result["errors"]


def test_only_explicit_own_wait_supervisor_is_read_only():
    own = "/root/workspace/pdblend-next-v1/campaign/A14B-sharegpt-slo90-v1/supervisor.py"
    assert d.classify_process(["python3", own, "--wait-only"]) == "read_only"
    assert d.classify_process(["python3", own]) == "gpu_work"
    assert d.classify_process(["python3", own, "--wait-only", "--run"]) == "gpu_work"
    assert d.classify_process(["python3", "campaign/other/supervisor.py", "--wait-only"]) == "gpu_work"


def test_owned_claim_requires_real_matching_exclusive_lock_and_exact_ancestry(tmp_path):
    lock = tmp_path / "test.lock"
    with lock.open("w") as stream:
        with pytest.raises(ValueError, match="no actual exclusive"):
            d.verify_owned_fd(stream.fileno(), lock)
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert d.verify_owned_fd(stream.fileno(), lock)["verified"] is True
        wrong = tmp_path / "wrong.lock"
        wrong.touch()
        with pytest.raises(ValueError, match="another file"):
            d.verify_owned_fd(stream.fileno(), wrong)
    ancestry = d.current_ancestry()
    assert d.owned_exclusions(ancestry[:1], ancestry)
    stale = copy.deepcopy(ancestry[:1])
    stale[0]["start_ticks"] += 1
    with pytest.raises(ValueError, match="exact current process"):
        d.owned_exclusions(stale, ancestry)


def test_long_gap_and_nonfinite_or_missing_identity_cannot_supply_stability():
    state = d.observe(d.new_state(), "C", evidence(observed=0), received_s=0)
    state = d.observe(state, "C", evidence(observed=10), received_s=10)
    state = d.observe(state, "C", evidence(observed=200), received_s=200)
    assert state["hosts"]["C"]["count"] == 1
    bad = evidence(observed=float("nan"))
    state = d.observe(state, "C", bad, received_s=210)
    assert state["hosts"]["C"]["count"] == 0
    bad = evidence(observed=220)
    bad["gpu"][0].pop("uuid")
    state = d.observe(state, "C", bad, received_s=220)
    assert state["hosts"]["C"]["count"] == 0
