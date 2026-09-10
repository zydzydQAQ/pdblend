"""Pure capacity-search state machine; no measurements or retries are invented."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from . import protocol as p


def new_search(*, system, dataset, n_gpus, initial_rate_rps=0.25, relative_width=0.05):
    row = p.validate_row(dict(system=system, dataset=dataset, n_gpus=n_gpus,
                              seed=p.PILOT_SEED, stage="pilot", rate_rps=initial_rate_rps))
    if relative_width != 0.05:
        raise ValueError("the predeclared capacity interval width is 5%")
    return dict(schema=1, protocol_id=p.PROTOCOL_ID, protocol_sha256=p.protocol_hash(),
                system=system, dataset=dataset, n_gpus=n_gpus,
                initial_rate_rps=row["rate_rps"], relative_width=relative_width,
                lower_rate_rps=None, upper_rate_rps=None, observations=[],
                status="pilot", blocked_reason=None, confirmed=False)


def _width(state):
    lo, hi = state["lower_rate_rps"], state["upper_rate_rps"]
    return None if lo is None or hi is None else (hi - lo) / lo


def formal_seed_brackets(state):
    """Derive each seed's complete history without replacing unfavorable points.

    The pilot bracket is only the starting proposal. Formal seeds measure both
    endpoints independently, then expand or bisect their own observed bracket.
    A failure below a success is non-monotonic evidence, never a valid bracket.
    """
    result = {}
    for seed in p.FORMAL_SEEDS:
        observations = [o for o in state["observations"]
                        if o["stage"] == "capacity" and o["seed"] == seed]
        initial = {boundary: any(o["boundary"] == boundary and o["rate_rps"] == state[key]
                                for o in observations)
                   for boundary, key in (("lower", "lower_rate_rps"), ("upper", "upper_rate_rps"))}
        passed = [o["rate_rps"] for o in observations if o["measurement_valid"] and o["capacity_pass"]]
        failed = [o["rate_rps"] for o in observations if o["measurement_valid"] and not o["capacity_pass"]]
        lo, hi = max(passed, default=None), min(failed, default=None)
        inconsistent = lo is not None and hi is not None and lo >= hi
        width = (hi - lo) / lo if lo is not None and hi is not None and not inconsistent else None
        initial_complete = all(initial.values())
        complete = initial_complete and width is not None and width <= state["relative_width"] + 1e-12
        next_rate = (hi / 2 if lo is None and hi is not None else lo * 2 if hi is None and lo is not None
                     else (lo + hi) / 2 if lo is not None and hi is not None else None)
        exhausted = (initial_complete and not complete and not inconsistent and
                     (next_rate is None or not math.isfinite(next_rate) or next_rate <= 0
                      or next_rate in {o["rate_rps"] for o in observations}))
        result[str(seed)] = dict(seed=seed, lower_rate_rps=lo, upper_rate_rps=hi,
            relative_width=width, initial_endpoints_measured=initial_complete,
            observation_count=len(observations),
            status="inconclusive" if inconsistent or exhausted else "complete" if complete else "searching",
            reason="non-monotonic formal observations" if inconsistent else
                   "floating-point search resolution exhausted" if exhausted else None,
            next_rate_rps=None if complete or inconsistent or exhausted else next_rate)
    return result


def next_action(state):
    """Return one exact next observation or a terminal status, without mutation."""
    status = state["status"]
    if status in ("blocked", "complete", "confirmation_failed"):
        return dict(action=status, reason=state.get("blocked_reason"),
                    lower_rate_rps=state["lower_rate_rps"], upper_rate_rps=state["upper_rate_rps"],
                    confirmed=state["confirmed"])
    lo, hi = state["lower_rate_rps"], state["upper_rate_rps"]
    common = dict(action="measure", system=state["system"], dataset=state["dataset"], n_gpus=state["n_gpus"])
    if status == "pilot":
        rate = (state["initial_rate_rps"] if lo is None and hi is None else
                hi / 2 if lo is None else lo * 2 if hi is None else (lo + hi) / 2)
        return dict(common, stage="pilot", seed=p.PILOT_SEED, rate_rps=rate, boundary=None)
    observed = {(o["boundary"], o["seed"], o["rate_rps"]) for o in state["observations"]
                if o["stage"] == "capacity"}
    for boundary, rate in (("lower", lo), ("upper", hi)):
        for seed in p.FORMAL_SEEDS:
            if (boundary, seed, rate) not in observed:
                return dict(common, stage="capacity", seed=seed, rate_rps=rate, boundary=boundary)
    for seed in p.FORMAL_SEEDS:
        bracket = formal_seed_brackets(state)[str(seed)]
        if bracket["status"] == "searching":
            return dict(common, stage="capacity", seed=seed, rate_rps=bracket["next_rate_rps"], boundary="refine")
    raise ValueError("confirmation state is exhausted but was not finalized")


def record_observation(state, observation):
    """Append an outcome; invalid engineering work blocks instead of saturating.

    All five independent seeds first retain measurements at both pilot endpoints.
    A shifted monotonic boundary is then refined independently for that seed;
    both favorable and unfavorable endpoint/refinement outcomes remain visible.
    Contradictory non-monotonic evidence makes that seed inconclusive. A claim
    is confirmed only once all five seed brackets are complete within 5%.
    """
    action = next_action(state)
    if action["action"] != "measure":
        raise ValueError("search is terminal; no automatic retries or replacement")
    obs = copy.deepcopy(observation)
    for key in ("system", "dataset", "n_gpus", "stage", "seed", "rate_rps", "boundary"):
        if key in obs and obs[key] != action[key]:
            raise ValueError("observation does not match next action: " + key)
        obs[key] = action[key]
    if type(obs.get("measurement_valid")) is not bool:
        raise ValueError("measurement_valid must be explicit")
    if obs["measurement_valid"] and type(obs.get("capacity_pass")) is not bool:
        raise ValueError("valid observations require an audited capacity_pass boolean")
    result = copy.deepcopy(state)
    result["observations"].append(obs)
    if not obs["measurement_valid"]:
        result.update(status="blocked", blocked_reason=obs.get("error", "engineering measurement invalid"))
        return result
    if obs["stage"] == "pilot":
        if obs["capacity_pass"]:
            result["lower_rate_rps"] = obs["rate_rps"]
        else:
            result["upper_rate_rps"] = obs["rate_rps"]
        width = _width(result)
        if width is not None and width <= result["relative_width"] + 1e-12:
            # LongBench N3 is explicitly a pilot-only normalization cell.
            result["status"] = "confirm" if result["n_gpus"] in p.FORMAL_SCALES[result["dataset"]] else "complete"
            result["pilot_interval_complete"] = True
    else:
        brackets = formal_seed_brackets(result)
        result["formal_seed_brackets"] = brackets
        if all(b["initial_endpoints_measured"] and b["status"] in ("complete", "inconclusive")
               for b in brackets.values()):
            good = all(b["status"] == "complete" for b in brackets.values())
            result.update(status="complete" if good else "confirmation_failed", confirmed=good,
                          blocked_reason=None if good else "one or more formal seeds have inconclusive capacity evidence")
    return result


def pilot_capacity_row(state):
    """Expose a conservative pilot-only lower bound with its whole search log."""
    width = _width(state)
    if width is None or width > state["relative_width"] + 1e-12 or not state.get("pilot_interval_complete"):
        raise ValueError("pilot has not established a <=5% pass/fail interval")
    return dict(system=state["system"], dataset=state["dataset"], n_gpus=state["n_gpus"],
                seed=p.PILOT_SEED, stage="pilot", capacity_rps=state["lower_rate_rps"],
                upper_rate_rps=state["upper_rate_rps"], measurement_valid=True,
                pilot_interval_complete=True, search_sha256=p.digest(state))


def weak_load_q(pilot_capacity_rows, *, dataset):
    if dataset not in p.SLOS:
        raise ValueError("unknown dataset")
    expected = {(system, n) for system in p.SYSTEMS for n in (3, 4)}
    capacities = {}
    for row in pilot_capacity_rows:
        if row.get("dataset") != dataset:
            continue
        key = (row.get("system"), row.get("n_gpus"))
        if key not in expected:
            continue
        if key in capacities:
            raise ValueError("duplicate pilot capacity; no favorable source selection")
        if (row.get("stage") != "pilot" or row.get("seed") != p.PILOT_SEED
                or row.get("measurement_valid") is not True or row.get("pilot_interval_complete") is not True):
            raise ValueError("q needs independently declared valid pilot capacity rows")
        capacities[key] = p._positive_number(row.get("capacity_rps"), "pilot capacity") / key[1]
    if set(capacities) != expected:
        raise ValueError("q needs all three systems at N=3,4 for this same dataset")
    return 0.7 * min(capacities.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("new")
    start.add_argument("--system", choices=p.SYSTEMS, required=True)
    start.add_argument("--dataset", choices=p.SLOS, required=True)
    start.add_argument("--n-gpus", type=int, required=True)
    start.add_argument("--initial-rate-rps", type=float, default=0.25)
    start.add_argument("--out", type=Path, required=True)
    step = sub.add_parser("next")
    step.add_argument("state", type=Path)
    record = sub.add_parser("record")
    record.add_argument("state", type=Path)
    record.add_argument("observation", type=Path)
    record.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "next":
        result = next_action(json.loads(args.state.read_text()))
        print(p.canonical_bytes(result).decode(), end="")
        return
    if args.command == "new":
        result = new_search(system=args.system, dataset=args.dataset, n_gpus=args.n_gpus,
                            initial_rate_rps=args.initial_rate_rps)
    else:
        result = record_observation(json.loads(args.state.read_text()), json.loads(args.observation.read_text()))
    with args.out.open("xb") as handle:
        handle.write(p.canonical_bytes(result))


if __name__ == "__main__":
    main()
