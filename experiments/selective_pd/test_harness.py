from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from .loadgen import RequestTrace, percentile, summarize_traces
from .power_sampler import integrate_energy


class PowerIntegrationTest(unittest.TestCase):
    def test_constant_power(self) -> None:
        samples = [
            {
                "gpu_index": 0,
                "gpu_uuid": "gpu-0",
                "monotonic_ns": timestamp,
                "power_w": 100.0,
                "gpu_util_pct": 50,
                "sm_clock_mhz": 900,
            }
            for timestamp in (0, 1_000_000_000, 2_000_000_000)
        ]
        result = integrate_energy(
            samples, 250_000_000, 1_750_000_000
        )
        self.assertAlmostEqual(result["total_energy_j"], 150.0)
        self.assertEqual(
            result["per_gpu"]["0"]["median_busy_sm_clock_mhz"], 900
        )


class LatencySummaryTest(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 90), 9.0)

    def test_trace_summary(self) -> None:
        trace = RequestTrace(
            request_id=0,
            scheduled_ns=0,
            started_ns=0,
            finished_ns=30_000_000,
            input_tokens=10,
            requested_output_tokens=2,
            output_tokens=2,
            token_times_ns=[10_000_000, 30_000_000],
            status=200,
            error=None,
        )
        summary = summarize_traces([trace])
        self.assertEqual(summary["p90_ttft_ms"], 10.0)
        self.assertEqual(summary["p90_tbt_ms"], 20.0)
        self.assertEqual(summary["output_tokens"], 2)


class PilotConfigTest(unittest.TestCase):
    def test_case_names_are_unique_per_suite(self) -> None:
        config_path = (
            Path(__file__).parent / "configs" / "pilot.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for suite in config["suites"].values():
            names = [case["name"] for case in suite["cases"]]
            self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
