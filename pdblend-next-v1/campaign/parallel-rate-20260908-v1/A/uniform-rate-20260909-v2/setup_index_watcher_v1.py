"""Publish immutable references to terminal setup windows outside measured trees."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def active(state):
    pid = state.get("pid")
    if not pid:
        return False
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return str(fields[19]) == str(state.get("startticks")) and fields[0] != "Z"


def exclusive_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("fresh external metadata output required")
    if args.out == args.stage or args.stage in args.out.parents:
        raise ValueError("metadata must remain outside all measured evidence trees")
    while True:
        source = args.stage / "status.json"
        if source.exists():
            state = read(source)
            if state.get("finished_s") and not active(state):
                break
        time.sleep(30)
    if state.get("node_lease_held"):
        raise ValueError("terminal setup still claims GPU lease")
    stages = []
    for label, relative in (("baseline_heterogeneous_native", "qualification/native"),
                            ("baseline_heterogeneous_candidate_frequency", "qualification/frequency")):
        directory = args.stage / relative
        terminal = directory / "status.json"
        if not terminal.exists():
            continue
        measured = read(terminal)
        if not measured.get("finished_s") or active(measured):
            raise ValueError("nested measurement has not terminated")
        if "measurement_end_s" not in measured:
            continue
        stages.append(dict(
            stage=label, terminal=ref(terminal),
            measurement_start_s=measured["measurement_start_s"],
            measurement_end_s=measured["measurement_end_s"],
            full_operation_energy_j=measured.get("full_operation_energy_j"),
            physical_measurement_valid=bool(measured.get("measurement_valid")),
            preparation_step_passed=bool(measured.get("passed", measured.get("complete"))),
            raw_power=ref(directory / "power/power.csv"),
            raw_utilization=ref(directory / "power/power.csv"),
            raw_clocks=ref(directory / "power/clocks.csv"),
            raw_column_convention="power.csv contains gpu0..7_w and gpu0..7_util_pct"))
    ordered = sorted(stages, key=lambda item: item["measurement_start_s"])
    for first, second in zip(ordered, ordered[1:]):
        if first["measurement_end_s"] > second["measurement_start_s"]:
            raise ValueError("setup windows overlap")
    args.out.mkdir(parents=True)
    index = args.out / "setup-energy-index.json"
    exclusive_json(index, dict(
        schema="new-A-baseline-preparation-energy-index-v1", node="Anew20260909", model="14b",
        stages=stages, owner_terminal=ref(source), producer=ref(__file__),
        accounting="Distinct terminal setup windows; preserve failed steps. Raw power remains authoritative. Do not add setup to formal rate energy. Deployment receipt is accounted separately."))
    exclusive_json(args.out / "metadata.json", dict(
        schema="uniform-v2-independent-evidence-metadata-v1", node="Anew20260909", model="14b",
        setup_energy_index=ref(index), evidence_closures=[], created_s=time.time()))


if __name__ == "__main__":
    main()
