"""Pure declaration and stopping rules for the independent ShareGPT sweep."""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation, localcontext
import math

PROTOCOL = "a14b-sharegpt-slo90-v1"
MODEL = "14b"
DATASET = "sharegpt"
SYSTEMS = ("pdblend", "mixed", "distserve", "dynamollm", "ecoserve")
BASELINES = SYSTEMS[1:]
SCALES = (0.5, 2.0)
SEED = 701
WINDOW_S = 100.0
REQUEST_TIMEOUT_S = 120.0
DRAIN_S = 120.0
INITIAL_RATES = ("0.4", "0.6", "0.8", "1", "1.25", "1.5", "2", "2.5", "3", "4", "5", "6")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool), "boolean is not a numeric protocol value")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid decimal protocol value") from exc
    require(decimal.is_finite() and decimal > 0, "positive finite protocol value required")
    return format(decimal, "f").rstrip("0").rstrip(".") if decimal.as_tuple().exponent < 0 else format(decimal, "f")


def scale_key(scale):
    key = number(scale)
    require(key in ("0.5", "2"), "only scales 0.5 and 2 are declared")
    return key


def effective_slo(scale):
    factor = Decimal(scale_key(scale))
    return dict(ttft_s=float(Decimal("5") * factor),
                tpot_s=float(Decimal("0.15") * factor), attainment_target=0.9)


def rate_at(index):
    require(type(index) is int and index >= 0, "rate index must be a nonnegative integer")
    if index < len(INITIAL_RATES):
        return INITIAL_RATES[index]
    exponent = index - len(INITIAL_RATES) + 1
    # Exact decimals avoid rate identities changing with the caller's context.
    with localcontext() as context:
        context.prec = max(32, 2 * exponent + 8)
        return number(Decimal("6") * Decimal("1.5") ** exponent)


def declaration():
    return dict(schema=1, protocol_id=PROTOCOL, model=MODEL, datasets=[DATASET],
                comparison_systems=list(SYSTEMS), slo_scales=list(SCALES),
                dataset_slo_s={DATASET: dict(ttft=5.0, tpot=0.15)},
                slo_attainment_target=0.9, arrival_seeds=[SEED], arrival_window_s=WINDOW_S,
                request_hard_timeout_s=REQUEST_TIMEOUT_S, drain_after_arrival_window_s=DRAIN_S,
                initial_rates_rps_decimal=list(INITIAL_RATES), extension_factor_decimal="1.5",
                stop_rule="Each scale stops after the first technically valid PDB point with 10*good < 9*offered; equality continues.",
                baseline_rule="All four baselines measure every valid PDB rate through and including that scale's first failing point.",
                stopping_is_saturation=False, scale_one_required=False,
                split="development", measurement_schema=3, execute_baselines=True,
                formal_eligible=False, statistical_scope="single arrival realization; no independent-seed confidence interval",
                energy_scope="all eight GPU boards, full 100-second window and observed native drain, including idle and failed work",
                deadline_s=None, continuation_scope="continues across the historical deadline; execution owner supplies operational limits",
                scheduling="first_idle_of_A_B_C", no_interrupt_active_campaigns=True,
                whole_campaign_single_host=True, execution_host=None,
                host_binding_rule="choose an actually idle A/B/C host; then freeze that host and its actual runtime binding")


def new_state():
    return dict(schema=1, protocol_id=PROTOCOL, scales={
        scale_key(scale): dict(status="active", next_index=0, endpoint_rate=None,
                               observations=[], technical_attempts=[])
        for scale in SCALES})


def _track(state, scale):
    require(state.get("schema") == 1 and state.get("protocol_id") == PROTOCOL,
            "state belongs to another protocol")
    return state["scales"][scale_key(scale)]


def next_rate(state, scale):
    track = _track(state, scale)
    return rate_at(track["next_index"]) if track["status"] == "active" else None


def _technical_errors(summary, technical_valid):
    errors = []
    if technical_valid is not True:
        errors.append("caller has not verified raw output, open-loop dispatch and native/outer cleanup")
    for key in ("measurement_valid", "fixed_window_valid"):
        if summary.get(key) is not True:
            errors.append(key + " is not true")
    for key in ("runtime_error", "sampling_error", "incomplete_drain", "technical_error"):
        if summary.get(key):
            errors.append(key)
    cleanup = summary.get("post_measurement_cleanup", {})
    if cleanup.get("cleanup_complete") is not True:
        errors.append("post-measurement cleanup is incomplete")
    # HTTP503 can carry an explicitly marked capacity admission refusal. Its
    # status alone is not an engineering failure; the raw auditor passed via
    # technical_valid distinguishes that case from unclassified server errors.
    if summary.get("http_503_engineering_count", 0) or summary.get("engineering_failure_count", 0):
        errors.append("engineering failures cannot establish a PDB endpoint")
    return errors


def record_pdb(state, scale, rate, summary, *, technical_valid=False, evidence=None):
    """Return a new state; never change the caller's state or certify raw data.

    The caller MUST pass technical_valid=True only after auditing all offered
    rows, exact workload/output and independent arrival evidence, and complete
    native plus outer cleanup. Unclassified HTTP503/engineering errors make
    that proof false; explicitly marked admission refusals may also use HTTP503.
    Valid admission refusals and hard timeouts remain in the offered denominator.
    A blocked track can only be advanced by an explicitly reviewed same-rate
    replacement observation; this function never launches an automatic retry.
    """
    result = copy.deepcopy(state)
    track = _track(result, scale)
    require(track["status"] in ("active", "blocked"), "PDB scale has already reached its endpoint")
    key = number(rate)
    require(key == rate_at(track["next_index"]), "PDB rates must be observed in their declared order")
    errors = _technical_errors(summary, technical_valid)
    offered, good = summary.get("offered_requests"), summary.get("good_requests")
    if type(offered) is not int or offered < 1 or type(good) is not int or not 0 <= good <= offered:
        errors.append("valid integer offered/good counts are required")
    if not errors:
        reported = summary.get("slo_attainment")
        if type(reported) not in (float, int) or not math.isfinite(reported) or not math.isclose(reported, good / offered, rel_tol=0, abs_tol=1e-12):
            errors.append("reported joint SLO differs from the complete offered denominator")
    if errors:
        track["technical_attempts"].append(dict(rate_rps_decimal=key, errors=errors,
                                                  evidence=copy.deepcopy(evidence)))
        track["status"] = "blocked"
        return result
    failed = good * 10 < offered * 9
    track["observations"].append(dict(rate_rps_decimal=key, good_requests=good,
                                      offered_requests=offered, slo_attainment=good / offered,
                                      below_target=failed, evidence=copy.deepcopy(evidence)))
    track["next_index"] += 1
    track["status"] = "stopped" if failed else "active"
    track["endpoint_rate"] = key if failed else None
    return result


def required_baseline_rates(state, scale):
    """Return only the valid PDB prefix, including its failing endpoint."""
    return [point["rate_rps_decimal"] for point in _track(state, scale)["observations"]]


def pdb_complete(state):
    return all(_track(state, scale)["status"] == "stopped" for scale in SCALES)
