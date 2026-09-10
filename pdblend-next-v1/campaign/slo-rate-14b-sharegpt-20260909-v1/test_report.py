import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("slo_rate_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def observation(self, host="A", rate=0.25, repeat=1, system="pdblend", **changes):
        cell = f"fixture-{host}-{system}-r{rate}-repeat{repeat}"
        directory = self.root / host / "measurements" / cell
        directory.mkdir(parents=True, exist_ok=True)
        raw = directory / "bench.csv"
        raw.write_text("fixture,raw\n1,2\n")
        power = directory / "power.csv"
        power.write_text("fixture,power\n1,8\n")
        checkpoint = directory / "checkpoint.json"
        report.save(checkpoint, {"artifacts": {str(raw): report.sha(raw), str(power): report.sha(power)}})
        value = dict(cell_id=cell, model="14b", dataset="sharegpt", measurement_host=host,
            system=system, rate_rps=rate, repeat=repeat, slo_scale=report.HOST_SCALES[host], seed=701,
            arrival_window_s=100, energy_measured_gpu_count=8, measurement_valid=True,
            independently_recomputed=True, work_complete=True, slo_attainment=1.0,
            strict_slo_recomputed=True, service_terminal_valid=True, actual_slo_config_verified=True,
            capacity_failures_independently_audited=True,
            energy_j=800, energy_per_good_request_j=100, good_requests=8, n_expected=8,
            completed_work_requests=8, gpu_util=0.25, ttft_avg_s=0.8, tpot_avg_s=0.1,
            goodput_measurement_rps=0.08, checkpoint=report.ref(checkpoint),
            raw_requests=report.ref(raw), raw_power=report.ref(power))
        value.update(changes)
        report.save(directory / "audited.json", {"observations": [value]})
        return value, directory

    def test_missing_metrics_are_null_and_csv_empty(self):
        self.observation(ttft_avg_s=None, tpot_avg_s=float("nan"), good_requests=0,
                         energy_per_good_request_j=None)
        result = report.build(self.root, plots=False)
        row = result["observations"][0]
        self.assertTrue(row["report_eligible"])
        self.assertIsNone(row["ttft_avg_s"])
        self.assertIsNone(row["tpot_avg_s"])
        self.assertIsNone(row["energy_per_good_request_j"])
        import csv
        with (self.root / "reports/current/observations.csv").open() as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved["energy_per_good_request_j"], "")

    def test_boundary_keeps_failure_when_repeat_passes(self):
        self.observation(rate=0.25)
        self.observation(rate=0.5, slo_attainment=0.89)
        self.observation(rate=0.5, repeat=2, slo_attainment=1.0)
        result = report.build(self.root, plots=False)
        self.assertEqual(result["groups"][0]["first_complete_slo_loss_rps"], 0.5)
        self.assertTrue(result["groups"][0]["boundary_repeat_observed"])
        self.assertEqual(len(result["observations"]), 3)

    def test_exact_90_percent_is_not_loss(self):
        self.observation(slo_attainment=0.9)
        result = report.build(self.root, plots=False)
        self.assertIsNone(result["groups"][0]["first_slo_loss_rps"])

    def test_incomplete_slo_loss_separate_from_complete_boundary(self):
        self.observation(rate=0.25, slo_attainment=0.1, work_complete=False)
        self.observation(rate=0.5, slo_attainment=0.8)
        group = report.build(self.root, plots=False)["groups"][0]
        self.assertEqual(group["first_slo_loss_rps"], 0.25)
        self.assertEqual(group["first_complete_slo_loss_rps"], 0.5)

    def test_changed_raw_file_invalidates_report_eligibility(self):
        _, directory = self.observation()
        (directory / "bench.csv").write_text("changed\n")
        row = report.build(self.root, plots=False)["observations"][0]
        self.assertEqual(row["source_integrity"], "mismatch")
        self.assertFalse(row["report_eligible"])
        self.assertEqual(row["energy_j"], 800)

    def test_missing_raw_reference_and_host_mismatch_are_not_eligible(self):
        self.observation(measurement_host="C", raw_requests=None)
        row = report.build(self.root, plots=False)["observations"][0]
        self.assertFalse(row["report_eligible"])
        self.assertIn("physical_host_mismatch", row["report_issues"])
        self.assertIn("raw_requests_reference_missing", row["report_issues"])

    def test_missing_historical_context_is_listed_without_erasing_raw_measurement(self):
        self.observation(original_declaration={"path": str(self.root / "old-missing.json"), "sha256": "a" * 64})
        row = report.build(self.root, plots=False)["observations"][0]
        self.assertEqual(row["source_integrity"], "missing")
        self.assertEqual(row["metric_source_integrity"], "verified")
        self.assertTrue(row["report_eligible"])

    def test_duplicate_rows_are_idempotent_but_conflicts_rejected(self):
        row, directory = self.observation()
        row.update(audit_reference=report.ref(directory / "audited.json"), engineering_attempt=1)
        report.save(self.root / "A/observations.json", [row])
        self.assertEqual(len(report.collect(self.root)[0]), 1)
        row["slo_attainment"] = 0.5
        report.save(self.root / "A/observations.json", [row])
        with self.assertRaisesRegex(ValueError, "conflicting observations"):
            report.collect(self.root)

    def test_qualification_document_does_not_imply_inventory_mirrored(self):
        qualification = self.root / "qualification.json"
        raw = self.root / "unmirrored-qualification-raw.jsonl"
        report.save(qualification, {"files": {str(raw): "b" * 64}})
        self.observation(qualification=report.ref(qualification))
        result = report.build(self.root, plots=False)
        row = result["observations"][0]
        self.assertEqual(row["qualification_mirror_status"], "verified")
        self.assertEqual(row["qualification_inventory_status"], "missing")
        self.assertFalse(row["local_context_complete"])
        self.assertTrue(row["report_eligible"])
        self.assertIn(str(raw), [item["path"] for item in report.read(self.root / "reports/current/raw-hash-index.json")])

    def test_B_reference_is_filtered_hash_pinned_and_immutable(self):
        row, _ = self.observation(host="B")
        source = self.root / "old-results.json"
        report.save(source, {"observations": [row, dict(row, cell_id="wrong-model", model="7b")]})
        before = source.read_bytes()
        report.save(self.root / "B/reference.json", {"source": report.ref(source)})
        result = report.build(self.root, plots=False)
        self.assertEqual(len(result["observations"]), 1)
        self.assertTrue(result["observations"][0]["reference_only"])
        self.assertEqual(source.read_bytes(), before)
        source.write_text("{}")
        with self.assertRaisesRegex(ValueError, "reference source hash changed"):
            report.collect(self.root)

    def test_setup_and_outer_ledger_never_added(self):
        self.observation(full_operation_energy_j=900)
        report.save(self.root / "A/setup-energy-ledger.json", {"entries": [
            {"energy_j": 1000, "full_operation_energy_j": 1100}]})
        result = report.build(self.root, plots=False)
        self.assertIsNone(result["energy_ledger"][0]["primary_plus_outer_total_j"])
        self.assertEqual(result["observations"][0]["energy_j"], 800)

    def test_three_host_plots_render_without_fake_missing_values(self):
        self.observation()
        report.build(self.root)
        output = self.root / "reports/current"
        self.assertEqual(len(list(output.glob("*-six-panel.png"))), 3)
        self.assertEqual(len(list(output.glob("*-six-panel.svg"))), 3)
        self.assertEqual(len(list(output.glob("*-six-panel.pdf"))), 3)

    def test_missing_and_ineligible_rates_break_actual_plot_segments(self):
        from matplotlib.path import Path as PlotPath
        points = [dict(rate_rps=.25, report_eligible=True, energy_j=100),
                  dict(rate_rps=.75, report_eligible=True, energy_j=300)]
        for series in (points, points + [dict(rate_rps=.5, report_eligible=False, energy_j=200)]):
            rates, values = report.grid_series(series, "energy_j")
            self.assertEqual(rates, [.25, .5, .75])
            self.assertTrue(math.isnan(values[1]))
            path = PlotPath(list(zip(rates, values)))
            codes = [code for _, code in path.iter_segments(remove_nans=True)]
            self.assertEqual(codes.count(PlotPath.MOVETO), 2)
            self.assertNotIn(PlotPath.LINETO, codes)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.production = Path(report.__file__).parent
        self.contract = report.load_local_module(self.production / "contract.py", "test_completion_contract")
        for rate in self.contract.grid("2"):
            directory = self.root / "workloads" / ("r" + rate)
            content_hash = report.hashlib.sha256(rate.encode()).hexdigest()
            report.save(directory / "trace.json", dict(content_pairing_sha256=content_hash, n_requests=2))
            trace_ref = report.ref(directory / "trace.json")
            workload = dict(campaign_id=self.contract.CAMPAIGN, model="14b", dataset="sharegpt",
                rate_rps=float(rate), seed=701, sampling_seed=20260907, arrival_window_s=100.,
                trace_reference=trace_ref, trace_sha256=trace_ref["sha256"], n_expected=2,
                content_pairing_sha256=content_hash)
            report.save(directory / "manifest.json", dict(workload_payload_sha256=
                report.hashlib.sha256(self.contract.encode(workload)).hexdigest()))
            workload["materialization_manifest"] = report.ref(directory / "manifest.json")
            report.save(directory / "workload.json", workload)
        self.rows, self.raw_by_node = [], {}
        self.checks = dict(checker=report.ref(self.production / "reports/crosscheck.py"), entries=[], issues=[])
        for node, cap in (("A", "1"), ("C", "2")):
            raw_rows = []
            for rate in self.contract.grid(cap):
                for system in self.contract.SYSTEMS:
                    repeats = (1, 2) if system == "pdblend" and rate == cap else (1,)
                    for repeat in repeats:
                        row = self.contract.make_row(node, rate, system, repeat, workload=self.workload(rate))
                        raw = dict(row, measurement_valid=True, independently_recomputed=True,
                            service_terminal_valid=True, strict_slo_recomputed=True, actual_slo_config_verified=True,
                            unknown_error_count=0, work_complete=True, energy_measured_gpu_count=8,
                            slo_attainment=.8 if system == "pdblend" and rate == cap else 1.)
                        path = self.root / node / "audits" / (row["cell_id"] + ".json")
                        report.save(path, raw)
                        reference = report.ref(path)
                        raw_rows.append(dict(raw, audit_reference=reference, engineering_attempt=1))
                        normalized = report.normalize(raw, node, reference)
                        normalized.update(audit_reference=reference, engineering_attempt=1,
                            report_eligible=True, metric_source_integrity="verified", local_context_complete=False)
                        self.rows.append(normalized)
                        self.checks["entries"].append(dict(cell_id=row["cell_id"], engineering_attempt=1,
                            status="passed", audit_reference=reference))
            self.raw_by_node[node] = raw_rows
            decision = self.contract.evaluate_group(node, raw_rows, materializer=self.workload)
            report.save(self.root / node / "run-002/status.json", dict(schema="slo-rate-node-status-v1",
                node=node, pid=999999999, startticks="0", started_s=1., finished_s=2., node_lease_held=False,
                complete=True, five_system_complete=True, phase="complete", decision=decision))

    def tearDown(self):
        self.temporary.cleanup()

    def workload(self, rate):
        return report.read(self.root / "workloads" / ("r" + self.contract.number(rate)) / "workload.json")

    def test_exact_grid_completes_while_full_context_remains_pending(self):
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertTrue(result["complete"])
        self.assertEqual(result["full_context_mirror_status"], "pending")
        self.assertEqual(result["nodes"]["A"]["observed_measurements"], 21)
        self.assertEqual(result["nodes"]["C"]["observed_measurements"], 41)
        self.assertEqual(result["nodes"]["A"]["pdb_first_loss_rps"], 1.)
        self.assertEqual(result["nodes"]["C"]["pdb_first_loss_rps"], 2.)

    def test_supervisor_complete_cannot_hide_missing_or_duplicate_measurements(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[-2]]):
            with self.subTest(count=len(rows)):
                result, _ = report.completion_status(self.root, rows, self.checks)
                self.assertFalse(result["complete"])
                self.assertFalse(result["nodes"]["C"]["complete"])

    def test_newer_running_supervisor_overrides_old_complete_status(self):
        report.save(self.root / "A/run-003/status.json", dict(schema="slo-rate-node-status-v1",
            node="A", pid=999999998, startticks="0", started_s=3., phase="measuring_mixed", complete=False))
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertFalse(result["complete"])
        self.assertEqual(result["nodes"]["A"]["supervisor_phase"], "measuring_mixed")

    def test_qualification_status_is_not_a_supervisor(self):
        report.save(self.root / "A/qualification/status.json", dict(schema="qualification-complete-v1",
            node="A", pid=999999998, started_s=99., phase="complete", complete=True))
        status, reference = report.latest_supervisor(self.root, "A")
        self.assertEqual(status["started_s"], 1.)
        self.assertIn("run-002", reference["path"])

    def test_changed_audit_or_false_aggregate_blocks_completion(self):
        self.rows[0]["unknown_error_count"] = 1
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertFalse(result["complete"])
        self.rows[0]["unknown_error_count"] = 0
        Path(self.rows[0]["audit_reference"]["path"]).write_text("{}")
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertFalse(result["complete"])

    def test_missing_failed_or_stale_crosscheck_blocks_completion(self):
        entry = self.checks["entries"][0]
        for replacement in (dict(entry, status="failed"),
                            dict(entry, audit_reference=dict(entry["audit_reference"], sha256="0" * 64)), None):
            with self.subTest(replacement=replacement):
                self.checks["entries"] = self.checks["entries"][1:]
                if replacement:
                    self.checks["entries"].insert(0, replacement)
                result, _ = report.completion_status(self.root, self.rows, self.checks)
                self.assertFalse(result["complete"])
                if replacement:
                    self.checks["entries"][0] = entry
                else:
                    self.checks["entries"].insert(0, entry)

    def test_ineligible_point_blocks_completion(self):
        self.rows[0]["report_eligible"] = False
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertFalse(result["complete"])

    def test_missing_workload_and_changed_manifest_keep_completion_pending(self):
        workload_path = self.root / "workloads/r2/workload.json"
        original = workload_path.read_bytes()
        workload_path.unlink()
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertFalse(result["complete"])
        self.assertFalse(workload_path.exists())
        workload_path.write_bytes(original)
        (self.root / "workloads/r2/manifest.json").write_text("{}")
        result, _ = report.completion_status(self.root, self.rows, self.checks)
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
