"""All-baseline dominance for already selected service-plus-tail observations.

This module does not select favorable baseline attempts or reuse qualification
from another revision. Baseline failures do not remove their recorded energy
from the user's deliberately stricter comparison target.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math


BASELINE_SYSTEMS = ("mixed", "distserve", "dynamollm", "ecoserve")
IDENTITY_FIELDS = ("trace_sha256", "seed", "duration_s", "slo_ttft_s", "slo_tpot_s",
                   "offered_requests", "offered_rps", "measurement_protocol_version",
                   "model_hash", "tokenizer_hash", "image_digest", "runtime_source_sha256",
                   "measurement_source_sha256", "gpu_uuids")
NUMERIC_IDENTITY_FIELDS = {"seed", "duration_s", "slo_ttft_s", "slo_tpot_s",
                           "offered_requests", "offered_rps"}
STANDARD_CONDITIONS = tuple((model, dataset, scale) for model in ('7B','14B','32B')
                            for dataset in ('alpaca','sharegpt','longbench') for scale in (.25,.5,.75,1.))
REPEAT_SCHEMA = 'pdblend-predeclared-paired-energy-repeats/v1'


def number(value):
    return float(value) if (isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value)) else None


def count(row, field):
    value = row.get(field)
    return value if type(value) is int and value >= 0 else None


def identity(row):
    source = row.get("comparison_identity") or {}
    result = {}
    for key in IDENTITY_FIELDS:
        value = source.get(key, row.get(key))
        if key in NUMERIC_IDENTITY_FIELDS:
            value = number(value)
        elif key == "gpu_uuids" and isinstance(value, list):
            value = sorted(value)
        result[key] = value
    return result


def scenario(row):
    return row.get("model"), row.get("dataset"), number(row.get("rate_scale")), number(row.get("offered_rps"))


def paired_errors(candidate, baseline, *, measurement_compatibility=()):
    errors = []
    if scenario(candidate) != scenario(baseline):
        errors.append("scenario_mismatch")
    left, right = identity(candidate), identity(baseline)
    for key in IDENTITY_FIELDS:
        if left[key] in (None, "", []) or right[key] in (None, "", []):
            errors.append("identity_missing:" + key)
        elif left[key] != right[key]:
            compatible = None
            if key == 'measurement_source_sha256' and measurement_compatibility:
                from .measurement_compatibility import compatible_pair
                if complete_energy(candidate) is not None and complete_energy(baseline) is not None:
                    compatible = compatible_pair(candidate, baseline, measurement_compatibility)
            if compatible is None:
                errors.append("identity_mismatch:" + key)
    for label, row, values in (("candidate", candidate, left), ("baseline", baseline, right)):
        for key in ("offered_requests", "offered_rps", "slo_ttft_s", "slo_tpot_s"):
            if number(row.get(key)) != values[key]:
                errors.append(label + "_identity_inconsistent:" + key)
    if candidate.get("repeat_id") != baseline.get("repeat_id"):
        errors.append("repeat_id_mismatch")
    return errors


def complete_energy(row):
    values = [number(row.get(key)) for key in ("total_energy_kj", "service_energy_kj", "tail_energy_kj")]
    if row.get("energy_measurement_complete") is not True or any(v is None or v < 0 for v in values):
        return None
    total, service, tail = values
    if not math.isclose(total, service + tail, rel_tol=1e-8, abs_tol=1e-5):
        return None
    return total


def measurement_qualification(row):
    # Legacy evidence_valid also includes profile/protocol gates. Its absence
    # or failure does not tell us whether the measurement-only gates passed.
    for field in ("measurement_qualified", "measurement_evidence_valid"):
        value = row.get(field)
        if isinstance(value, bool):
            return "pass" if value else "fail"
    return "unknown"


def combined_qualification(rows):
    values = [measurement_qualification(r) for r in rows]
    return "fail" if "fail" in values else "unknown" if "unknown" in values else "pass"


def compare_point(candidate, baselines, *, measurement_compatibility=()):
    """Compare one candidate to four *preselected* baseline observations."""
    reasons, missing, pairs = [], [], {}
    offered, successful, good = (count(candidate, k) for k in
                                ("offered_requests", "successful_requests", "joint_slo_requests"))
    counts_valid = offered is not None and offered > 0 and successful is not None and good is not None
    counts_valid = counts_valid and 0 <= good <= successful <= offered
    all_success = bool(counts_valid and successful == offered
                       and candidate.get("all_requests_successful") is True
                       and count(candidate, "failed_requests") == 0
                       and count(candidate, "unresolved_requests") == 0)
    if not all_success:
        reasons.append("candidate_failed_or_unresolved_requests")
    attainment = good / offered if counts_valid else None
    absolute_slo = attainment is not None and attainment >= .9
    for metric, limit in (("ttft_p99_s", "slo_ttft_s"), ("tpot_p99_s", "slo_tpot_s")):
        observed, allowed = number(candidate.get(metric)), number(candidate.get(limit))
        passed = observed is not None and allowed is not None and 0 <= observed <= allowed and allowed > 0
        absolute_slo = absolute_slo and passed
        if not passed:
            reasons.append("candidate_missing_or_exceeded:" + metric)
    if attainment is None or attainment < .9:
        reasons.append("candidate_joint_attainment_below_90_or_missing")
    candidate_energy = complete_energy(candidate)
    if candidate_energy is None:
        missing.append("candidate_incomplete_energy")
    for system in BASELINE_SYSTEMS:
        baseline = baselines.get(system)
        if baseline is None:
            missing.append(system + ":missing_or_ambiguous")
            continue
        errors = paired_errors(candidate, baseline, measurement_compatibility=measurement_compatibility)
        if errors:
            missing.extend(system + ":" + error for error in errors)
            continue
        baseline_good = count(baseline, "joint_slo_requests")
        baseline_offered = count(baseline, "offered_requests")
        if baseline_good is None or baseline_offered != offered or baseline_good > offered:
            missing.append(system + ":invalid_joint_request_count")
            baseline_good = None
        energy = complete_energy(baseline)
        if energy is None:
            missing.append(system + ":incomplete_energy")
        pairs[system] = dict(total_energy_kj=energy, joint_slo_requests=baseline_good,
                             baseline_slo_pass=baseline.get("slo_pass"),
                             baseline_all_requests_successful=baseline.get("all_requests_successful"),
                             revision=baseline.get("revision"), receipt_path=baseline.get("receipt_path"),
                             measurement_qualification=measurement_qualification(baseline),
                             formal_eligible=baseline.get("formal_eligible") is True)
        if identity(candidate)['measurement_source_sha256'] != identity(baseline)['measurement_source_sha256']:
            from .measurement_compatibility import compatible_pair
            pairs[system]['measurement_compatibility'] = compatible_pair(candidate, baseline, measurement_compatibility)
    complete_pairs = len(pairs) == len(BASELINE_SYSTEMS)
    energies = [p["total_energy_kj"] for p in pairs.values() if p["total_energy_kj"] is not None]
    joint_counts = [p["joint_slo_requests"] for p in pairs.values() if p["joint_slo_requests"] is not None]
    all_energy = complete_pairs and len(energies) == len(BASELINE_SYSTEMS) and candidate_energy is not None
    minimum = min(energies) if energies else None
    maximum_good = max(joint_counts) if joint_counts else None
    # A missing comparator prevents a win, but cannot erase a counterexample
    # that is already observed against another complete baseline.
    known_energy_loss = candidate_energy is not None and minimum is not None and candidate_energy >= minimum
    energy_win = candidate_energy < minimum if all_energy else False if known_energy_loss else None
    joint_win = good >= maximum_good if counts_valid and len(joint_counts) == len(BASELINE_SYSTEMS) else None
    if energy_win is False:
        reasons.append("candidate_not_strictly_below_every_baseline_energy")
    if joint_win is False:
        reasons.append("candidate_joint_good_below_best_baseline")
    energy_saving = 100 * (1 - candidate_energy / minimum) if all_energy and minimum > 0 else None
    available_saving = (100 * (1 - candidate_energy / minimum)
                        if candidate_energy is not None and minimum is not None and minimum > 0 else None)
    numerical_win = bool(all_success and absolute_slo and joint_win and energy_win) if not missing else None
    candidate_qualified = measurement_qualification(candidate)
    if numerical_win is True and candidate_qualified != 'pass':
        if candidate_qualified == 'unknown':
            missing.append('candidate_measurement_qualification_unknown')
        else:
            reasons.append('candidate_measurement_qualification_failed')
    observed_win = (numerical_win and candidate_qualified == 'pass') if not missing else None
    rows = [candidate] + [baselines[s] for s in BASELINE_SYSTEMS if baselines.get(s) is not None]
    return dict(series=candidate.get("series"), revision=candidate.get("revision"),
                model=candidate.get("model"), dataset=candidate.get("dataset"),
                rate_scale=candidate.get("rate_scale"), offered_rps=candidate.get("offered_rps"),
                point_id=candidate.get("point_id"), repeat_id=candidate.get("repeat_id"),
                receipt_path=candidate.get("receipt_path"), comparison_identity=identity(candidate),
                status="incomplete" if missing else "observed_win" if observed_win else "failed",
                observed_goal_met=observed_win, numerical_goal_met=numerical_win,
                all_requests_successful=all_success,
                absolute_slo_pass=bool(absolute_slo), joint_slo_requests=good,
                max_baseline_joint_slo_requests=maximum_good, joint_good_goal_met=joint_win,
                total_energy_kj=candidate_energy, min_available_baseline_energy_kj=minimum,
                energy_complete=bool(all_energy), energy_goal_met=energy_win,
                saving_vs_min_baseline_pct=energy_saving,
                saving_vs_min_available_baseline_pct=available_saving,
                measurement_qualification=measurement_qualification(candidate),
                comparison_measurement_qualification=combined_qualification(rows) if complete_pairs else "unknown",
                formal_eligible=candidate.get("formal_eligible") is True,
                comparison_formal_eligible=bool(complete_pairs and all(r.get("formal_eligible") is True for r in rows)),
                missing=missing, reasons=reasons, baselines=pairs)


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def analyze_points(points, *, measurement_compatibility=()):
    """Separate revisions; reject ambiguous attempts; require actual paired repeats."""
    baseline_index = defaultdict(list)
    candidates = []
    for row in points:
        if row.get("system") == "pdblend" or str(row.get("series", "")).startswith("pd_"):
            candidates.append(row)
        elif row.get("system") in BASELINE_SYSTEMS:
            baseline_index[(row["system"], _key(scenario(row)), _key(row.get("repeat_id")))].append(row)
    comparisons = []
    for candidate in candidates:
        selected = {}
        for system in BASELINE_SYSTEMS:
            rows = baseline_index[(system, _key(scenario(candidate)), _key(candidate.get("repeat_id")))]
            matches = [r for r in rows if not paired_errors(candidate, r,
                       measurement_compatibility=measurement_compatibility)]
            selected[system] = matches[0] if len(matches) == 1 else None
        comparisons.append(compare_point(candidate, selected, measurement_compatibility=measurement_compatibility))
    groups = defaultdict(list)
    for row in comparisons:
        key = (row["series"], row["revision"], row["model"], row["dataset"], row["rate_scale"],
               row["offered_rps"], _key(row["comparison_identity"]))
        groups[key].append(row)
    cases = []
    for rows in groups.values():
        ref = rows[0]
        repeated_ids = [r["repeat_id"] for r in rows]
        independent = (len(rows) >= 3 and all(v is not None for v in repeated_ids)
                       and len({_key(v) for v in repeated_ids}) == len(rows))
        for label in ("pdblend", *BASELINE_SYSTEMS):
            paths = [r["receipt_path"] if label == "pdblend" else r["baselines"].get(label, {}).get("receipt_path") for r in rows]
            independent = independent and all(paths) and len(set(paths)) == len(rows)
        margins = [r["saving_vs_min_baseline_pct"] for r in rows if r["saving_vs_min_baseline_pct"] is not None]
        incomplete = any(r["status"] == "incomplete" for r in rows)
        all_win = all(r["observed_goal_met"] is True for r in rows)
        small_margin = any(0 < margin < 3 for margin in margins)
        status = ("incomplete" if incomplete else "failed" if not all_win else
                  "small_margin_needs_3_paired_repeats" if small_margin and not independent else
                  "stable_observed_win" if independent else "observed_win")
        case = {k: ref[k] for k in ("series", "revision", "model", "dataset", "rate_scale", "offered_rps", "point_id")}
        case.update(status=status, observations=len(rows), independent_paired_repeats=bool(independent),
                    min_saving_pct=min(margins) if margins else None,
                    max_saving_pct=max(margins) if margins else None,
                    energy_complete=all(r["energy_complete"] for r in rows),
                    measurement_qualification=("fail" if any(r["comparison_measurement_qualification"] == "fail" for r in rows)
                        else "pass" if all(r["comparison_measurement_qualification"] == "pass" for r in rows) else "unknown"),
                    formal_eligible=all(r["comparison_formal_eligible"] for r in rows))
        cases.append(case)
        for row in rows:
            row["case_status"] = status
    return dict(schema="all_baseline_cohort_dominance/v2", energy_scope="eight_gpu_boards_service_plus_tail",
                measurement_compatibility=[r['manifest_binding'] for r in measurement_compatibility],
                baseline_selection="all_four_preselected_baselines_including_failed_requests",
                thresholds=dict(min_attainment=.9, small_energy_margin_pct=3., required_paired_repeats=3),
                comparisons=comparisons, cases=cases, status_counts=dict(Counter(c["status"] for c in cases)))


def condition(row):
    return row.get('model'), row.get('dataset'), number(row.get('rate_scale'))


def condition_id(key):
    return f'{key[0]}/{key[1]}/x{key[2]:g}'


def repeat_reference(row):
    return dict(system=row['system'],revision=row['revision'],point_id=row['point_id'],
        receipt_path=row['receipt_path'],receipt_sha256=row.get('receipt_sha256'),
        comparison_identity=identity(row))


def declare_paired_repeats(points, *, candidate_revision, declared_at_s, measurement_compatibility=()):
    """Fix all 36 conditions before three new, paired, same-trace experiments.

    Only complete observed wins with a strictly positive margin below 3% need
    repeats. Four baseline identities stay fixed, including failed baselines.
    """
    candidates=defaultdict(list);baselines=defaultdict(list)
    for row in points:
        key=condition(row)
        if row.get('system')=='pdblend' and row.get('revision')==candidate_revision:candidates[key].append(row)
        elif row.get('system') in BASELINE_SYSTEMS:baselines[(key,row['system'])].append(row)
    if set(candidates)!=set(STANDARD_CONDITIONS) or any(len(rows)!=1 for rows in candidates.values()):
        raise ValueError('exactly one candidate from the declared revision is required for each of the 36 original conditions')
    if number(declared_at_s) is None or declared_at_s<=0:raise ValueError('declaration timestamp required')
    cases=[]
    for key in STANDARD_CONDITIONS:
        candidate=candidates[key][0]
        if candidate.get('repeat_id') is not None:raise ValueError('initial snapshot cannot substitute an already selected repeat')
        fixed=identity(candidate)
        if fixed['seed']!=701 or fixed['duration_s']!=150 or len(set(fixed['gpu_uuids'] or []))!=8:
            raise ValueError('original seed 701, 150-second service and complete eight-GPU fleet required')
        selected={system:rows[0] if len(rows:=baselines[(key,system)])==1 else None for system in BASELINE_SYSTEMS}
        initial=compare_point(candidate,selected,measurement_compatibility=measurement_compatibility)
        margin=initial['saving_vs_min_baseline_pct']
        required=bool(initial['observed_goal_met'] is True and margin is not None and 0<margin<3.)
        cases.append(dict(condition_id=condition_id(key),condition=list(key),initial=initial,
            candidate=repeat_reference(candidate),baselines={system:repeat_reference(row) if row else None
                for system,row in selected.items()},repeat_required=required))
    return dict(schema=REPEAT_SCHEMA,declared_at_s=declared_at_s,candidate_revision=candidate_revision,
        repeat_ids=[1,2,3],conditions=cases,planned_observations=[],
        selection_rule='initial_complete_observed_win_and_0_lt_margin_lt_3_percent',
        acceptance_rule='all_exactly_three_predeclared_repeats_must_pass_no_retry_substitution',
        energy_scope='eight_gpu_boards_service_plus_tail',goal_complete=False)


def evaluate_declared_repeats(protocol, observations, *, measurement_compatibility=()):
    """Reject missing, reused, replaced or undeclared repeat evidence.

    This accepts canonical receipt-derived rows. Operational callers must also
    verify each row's receipt, point, source and frozen startup/config bindings.
    """
    if protocol.get('schema')!=REPEAT_SCHEMA or protocol.get('repeat_ids')!=[1,2,3]:
        raise ValueError('exactly the three predeclared repeat IDs 1, 2, 3 are required')
    cases=protocol['conditions'];keys=[tuple(row['condition']) for row in cases]
    if len(cases)!=36 or set(keys)!=set(STANDARD_CONDITIONS):raise ValueError('complete original 36-condition context required')
    expected={};slots=set();by_case={case['condition_id']:case for case in cases}
    for plan in protocol['planned_observations']:
        slot=(plan['condition_id'],plan['repeat_id'],plan['system'])
        case=by_case.get(plan['condition_id'])
        if (not case or not case['repeat_required'] or plan['repeat_id'] not in (1,2,3)
                or plan['system'] not in ('pdblend',*BASELINE_SYSTEMS) or slot in slots or plan['point_id'] in expected):
            raise ValueError('ambiguous or unexpected predeclared observation')
        reference=case['candidate'] if plan['system']=='pdblend' else case['baselines'][plan['system']]
        if plan['revision']!=reference['revision'] or plan['comparison_identity']!=reference['comparison_identity']:
            raise ValueError('repeat source or same-trace identity differs from fixed initial template')
        expected[plan['point_id']]=plan;slots.add(slot)
    required={(case['condition_id'],repeat,system) for case in cases if case['repeat_required']
              for repeat in (1,2,3) for system in ('pdblend',*BASELINE_SYSTEMS)}
    if slots!=required:raise ValueError('predeclaration must plan all 15 observations for every small-margin condition')
    indexed=defaultdict(list);diagnostics=[]
    for row in observations:
        if row.get('point_id') in expected:indexed[row['point_id']].append(row)
        else:diagnostics.append(dict(point_id=row.get('point_id'),reason='undeclared_diagnostic_not_a_substitute'))
    paths=Counter(row.get('receipt_path') for row in observations if row.get('point_id') in expected)
    hashes=Counter(row.get('receipt_sha256') for row in observations if row.get('point_id') in expected)
    selected={};invalid=[]
    for name,plan in expected.items():
        rows=indexed[name];reasons=[]
        if len(rows)!=1:reasons.append('missing_or_multiple_attempts_no_retry_selection')
        else:
            row=rows[0]
            if (row.get('system')!=plan['system'] or row.get('revision')!=plan['revision']
                    or row.get('repeat_id')!=plan['repeat_id'] or identity(row)!=plan['comparison_identity']
                    or condition_id(condition(row))!=plan['condition_id']):reasons.append('repeat_identity_differs')
            if not row.get('receipt_path') or not row.get('receipt_sha256') or paths[row.get('receipt_path')]!=1 or hashes[row.get('receipt_sha256')]!=1:
                reasons.append('receipt_not_independent')
            if number(row.get('service_start_s')) is None or row['service_start_s']<=protocol['declared_at_s']:
                reasons.append('measurement_not_after_predeclaration')
            if row.get('cleanup_passed') is not True:reasons.append('native_cleanup_unverified')
        if reasons:invalid.append(dict(point_id=name,reasons=reasons))
        else:selected[(plan['condition_id'],plan['repeat_id'],plan['system'])]=rows[0]
    result=[]
    for case in cases:
        initial=case['initial'];margin=initial.get('saving_vs_min_baseline_pct')
        required=bool(initial.get('observed_goal_met') is True and margin is not None and 0<margin<3.)
        if required!=case['repeat_required']:raise ValueError('small-margin repeat requirement was altered')
        repeated=[]
        if required:
            for repeat in (1,2,3):
                candidate=selected.get((case['condition_id'],repeat,'pdblend'))
                if candidate is None:
                    repeated.append(dict(repeat_id=repeat,status='incomplete',observed_goal_met=None));continue
                comparison=compare_point(candidate,{system:selected.get((case['condition_id'],repeat,system))
                    for system in BASELINE_SYSTEMS},measurement_compatibility=measurement_compatibility)
                # Repeating a baseline must never relax the user's already
                # observed target. Both frozen and contemporaneous bounds hold.
                frozen_good=initial['max_baseline_joint_slo_requests']
                frozen_energy=initial['min_available_baseline_energy_kj']
                good=count(candidate,'joint_slo_requests');energy=complete_energy(candidate)
                initial_pass=bool(good is not None and good>=frozen_good and energy is not None and energy<frozen_energy)
                comparison.update(paired_repeat_pass=comparison['observed_goal_met'] is True,
                    initial_target_pass=initial_pass,frozen_initial_joint_slo_requests=frozen_good,
                    frozen_initial_min_energy_kj=frozen_energy,
                    saving_vs_frozen_initial_min_pct=100*(1-energy/frozen_energy)
                        if energy is not None and frozen_energy>0 else None)
                if not initial_pass:
                    comparison['observed_goal_met']=False
                    comparison['numerical_goal_met']=False
                    if comparison['status']!='incomplete':comparison['status']='failed'
                    comparison['reasons'].append('candidate_must_also_beat_frozen_initial_energy_and_attainment_targets')
                repeated.append(comparison)
            passed=all(row['observed_goal_met'] is True for row in repeated)
            status=('stable_observed_win' if passed else 'incomplete' if any(row['status']=='incomplete' for row in repeated) else 'failed')
        else:passed=initial['observed_goal_met'] is True;status=initial['status']
        result.append(dict(condition_id=case['condition_id'],repeat_required=required,status=status,
                           criteria_met=passed,repeats=repeated,initial_status=initial['status']))
    return dict(schema='pdblend-predeclared-repeat-acceptance/v1',conditions=result,
        all_36_observed_criteria_met=all(row['criteria_met'] for row in result),
        status_counts=dict(Counter(row['status'] for row in result)),invalid_observations=invalid,
        diagnostics=diagnostics,goal_complete=False,requires_result_review=True)
