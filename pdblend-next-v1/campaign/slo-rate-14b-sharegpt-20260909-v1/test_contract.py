"""Meaningful CPU checks for boundary semantics, pairing, and source fidelity."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


c = module("slo_rate_test_contract", "contract.py")
g = module("slo_rate_test_generator", "generate.py")
REFERENCE = dict(path="/independent/audit.json", sha256="a" * 64)


def workload(rate):
    rate = c.number(rate)
    return dict(model="14b", dataset="sharegpt", rate_rps=float(rate),
                seed=701, sampling_seed=20260907, arrival_window_s=100.,
                trace_sha256=hashlib.sha256(rate.encode()).hexdigest(),
                content_pairing_sha256="c" * 64)


def observation(node="A", rate="0.25", q=.99, system="pdblend", repeat=1, **fields):
    row = c.make_row(node, rate, system, repeat, workload=workload(rate))
    result = dict(row, measurement_valid=True, service_terminal_valid=True,
        strict_slo_recomputed=True, independently_recomputed=True,
        unknown_error_count=0, audit_reference=REFERENCE,
        work_complete=True, slo_attainment=q)
    result.update(fields)
    return result


class BoundaryContractTests(unittest.TestCase):
    def decide(self, observations, rate="0.25", node="A"):
        return c.evaluate_rate(node, rate, observations, workload=workload(rate))

    def test_shared_trace_scaled_joint_thresholds_and_unique_cells(self):
        a = c.make_row("A", ".25", "pdblend", workload=workload(".25"))
        b = c.make_row("C", ".25", "pdblend", workload=workload(".25"))
        self.assertEqual((a["slo_ttft_s"], a["slo_tpot_s"]), (2.5, .075))
        self.assertEqual((b["slo_ttft_s"], b["slo_tpot_s"]), (10., .3))
        self.assertEqual(a["trace_sha256"], b["trace_sha256"])
        self.assertNotEqual(a["cell_id"], b["cell_id"])
        self.assertEqual(a["request_hard_timeout_s"], b["request_hard_timeout_s"])
        with self.assertRaises(ValueError):
            c.make_row("B", ".25", "mixed", workload=workload(".25"))
        with self.assertRaises(ValueError):
            c.cell_id("A", ".25", "mixed", 2)

    def test_exact_grid_no_artificial_upper_limit(self):
        for bad in (True, 0, -1, "NaN", "Infinity", ".3", "0.25000001"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c.number(bad)
        self.assertEqual(c.number("1000.25"), "1000.25")
        self.assertEqual(c.grid(".75"), ["0.25", "0.5", "0.75"])

    def test_equality_continues_but_first_loss_is_permanent(self):
        self.assertEqual(self.decide([observation(q=.90)])["status"], "advance")
        first = observation(q=.89)
        pending = self.decide([first])
        self.assertEqual(pending["status"], "confirm_boundary")
        self.assertTrue(pending["cap_observed"])
        confirmed = self.decide([first, observation(q=.97, repeat=2)])
        self.assertEqual(confirmed["status"], "cap_confirmed")
        self.assertTrue(confirmed["threshold_straddles"])
        self.assertFalse(confirmed["increase_rate_allowed"])

    def test_audited_capacity_refusal_or_timeout_is_scored(self):
        for failure in ("capacity_rejection", "declared_deadline"):
            for q, status in ((.89, "confirm_boundary"), (.95, "advance")):
                obs = observation(q=q, work_complete=False,
                    capacity_failures_independently_audited=True, failure_class=failure)
                self.assertEqual(self.decide([obs])["status"], status)
                self.assertFalse(obs["work_complete"])

    def test_unknown_error_cannot_be_upgraded_by_a_capacity_label(self):
        for extra in ({"unknown_error_count": 1}, {"service_terminal_valid": False},
                      {"measurement_valid": False}, {"independently_recomputed": False},
                      {"audit_reference": None}, {"capacity_failures_independently_audited": False}):
            obs = observation(q=.1, work_complete=False,
                              capacity_failures_independently_audited=True)
            obs.update(extra)
            with self.subTest(extra=extra):
                decision = self.decide([obs])
                self.assertEqual(decision["status"], "engineering_diagnosis")
                self.assertFalse(decision["cap_observed"])
                self.assertEqual(decision["next_tasks"], [])

    def test_one_diagnosed_replacement_and_no_valid_cherry_pick(self):
        invalid = observation(measurement_valid=False)
        retry = observation(engineering_attempt=2, repair_reference=REFERENCE)
        self.assertEqual(self.decide([invalid, retry])["status"], "advance")
        again = dict(retry, measurement_valid=False)
        self.assertEqual(self.decide([invalid, again])["status"], "blocked_engineering")
        with self.assertRaises(ValueError):
            self.decide([invalid, dict(retry, engineering_attempt=3)])
        with self.assertRaises(ValueError):
            self.decide([observation(q=.85), retry])
        with self.assertRaises(ValueError):
            self.decide([invalid, dict(retry, repair_reference=None)])

    def test_confirmation_fault_keeps_cap_and_does_not_schedule_forever(self):
        first = observation(q=.85)
        broken = observation(repeat=2, measurement_valid=False)
        result = self.decide([first, broken])
        self.assertTrue(result["cap_observed"])
        self.assertEqual(result["status"], "engineering_diagnosis")
        replacement = dict(broken, engineering_attempt=2, repair_reference=REFERENCE)
        result = self.decide([first, broken, replacement])
        self.assertTrue(result["cap_observed"])
        self.assertEqual(result["status"], "blocked_engineering")

    def test_confirmation_cannot_exist_before_a_first_loss(self):
        with self.assertRaises(ValueError):
            self.decide([observation(q=.95), observation(q=.85, repeat=2)])
        with self.assertRaises(ValueError):
            self.decide([observation(q=.85, repeat=2)])

    def test_baselines_include_first_grid_and_terminal_point(self):
        obs = [observation(rate=".25"), observation(rate=".5", q=.80),
               observation(rate=".5", q=.92, repeat=2)]
        result = c.evaluate_group("A", obs, materializer=workload)
        self.assertEqual(result["eligible_rates"], ["0.25", "0.5"])
        self.assertEqual(len(result["baseline_tasks"]), 8)
        self.assertFalse(result["complete"])
        for row in result["baseline_tasks"]:
            obs.append(observation(rate=row["rate_rps"], system=row["system"], q=.1,
                work_complete=False, capacity_failures_independently_audited=True))
        self.assertTrue(c.evaluate_group("A", obs, materializer=workload)["complete"])
        start_loss = c.evaluate_group("A", [observation(q=.1)], materializer=workload)
        self.assertEqual(start_loss["eligible_rates"], ["0.25"])
        self.assertEqual(len(start_loss["baseline_tasks"]), 4)

    def test_gap_cross_host_and_changed_trace_are_rejected(self):
        with self.assertRaises(ValueError):
            c.evaluate_group("A", [observation(rate=".5")], materializer=workload)
        with self.assertRaises(ValueError):
            c.evaluate_group("A", [observation(node="C")], materializer=workload)
        with self.assertRaises(ValueError):
            self.decide([observation(trace_sha256="f" * 64)])

    def test_next_rate_does_not_depend_on_baseline_performance(self):
        obs = [observation(q=.95), observation(system="ecoserve", q=.01)]
        result = c.evaluate_group("A", obs, materializer=workload)
        self.assertEqual(result["rate_rps"], .5)
        self.assertEqual(result["status"], "measure_pdblend")


class FrozenTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = g.freeze_inputs()

    def test_all_existing_rates_equal_frozen_B_trace_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            for rate, old in self.inputs["known_workloads"].items():
                generated = g.materialize_rate(rate, out_root=directory)
                self.assertEqual(generated["trace_sha256"], old["original_trace"]["sha256"])
                self.assertEqual(generated["content_pairing_sha256"], old["content_pairing_sha256"])
                self.assertEqual(generated, g.materialize_rate(rate, out_root=directory))

    def test_future_grid_can_generate_without_historical_loader_or_results(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(g, "_original_sources", side_effect=AssertionError("old files used")):
            generated = g.materialize_rate("2", out_root=directory)
            a = c.make_row("A", "2", "pdblend", workload=generated)
            b = c.make_row("C", "2", "ecoserve", workload=generated)
            self.assertEqual(a["trace_reference"], b["trace_reference"])
            self.assertEqual(a["content_pairing_sha256"], b["content_pairing_sha256"])
            self.assertGreater(generated["n_requests"], 0)
            trace = c.checked(generated["trace_reference"])
            self.assertTrue(all(0 <= q["arrival_s"] < 100 for q in trace["requests"]))
            self.assertEqual(trace["requests"][0]["arrival_s"], 0)

    def test_modified_cached_workload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = g.materialize_rate(".25", out_root=directory)
            index = Path(directory) / "r0.25/workload.json"
            generated["expected_generated_tokens"] += 1
            index.write_bytes(c.encode(generated))
            with self.assertRaises(ValueError):
                g.materialize_rate(".25", out_root=directory)


if __name__ == "__main__":
    unittest.main()
