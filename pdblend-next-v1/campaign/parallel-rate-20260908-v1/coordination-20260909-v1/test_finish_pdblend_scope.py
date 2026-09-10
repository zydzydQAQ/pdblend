"""CPU fixtures for switching from the five-system supervisor to PDB-only."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import finish_pdblend_scope as finish


class BoundarySwitch(unittest.TestCase):
    def run_fixture(self, all_capped):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            contract = root / "contract.py"
            contract.write_text(
                "def resolve_group(declaration, model, dataset, actual_host): return dataset\n"
                "def select_group(group, observations):\n"
                "    return {'phase': observations[0]['decision'], 'cap_rate_rps': 3.0}\n")
            plan = root / "plan.json"
            stop = root / "STOP"
            finish.write_new(plan, dict(node="fixture", model="7b",
                expected_hostname=socket.gethostname(), contract=finish.ref(contract),
                dataset_order=["alpaca", "sharegpt", "longbench"], stop_paths=[str(stop)]))
            observations = []
            for dataset in ("alpaca", "sharegpt", "longbench"):
                item = root / (dataset + ".json")
                decision = "pdblend" if not all_capped and dataset == "longbench" else "baselines"
                finish.write_new(item, dict(system="pdblend", dataset=dataset, decision=decision))
                observations.append(finish.ref(item))
            state = root / "status.json"
            finish.write_new(state, dict(plan=finish.ref(plan), node="fixture", model="7b",
                node_lease_held=False, observations=observations, declaration={}))
            priority = root / "priority.json"
            finish.write_new(priority, dict(effective_systems=["pdblend"],
                future_pdb_progress_must_not_wait_for_baseline_completion=True))
            out = root / "complete.json"
            argv = ["fixture", "--plan", str(plan), "--pipeline-status", str(state),
                    "--priority", str(priority), "--out", str(out), "--stop", str(stop)]
            with patch.object(sys, "argv", argv):
                if all_capped:
                    finish.main()
                    result = json.loads(out.read_text())
                    self.assertTrue(stop.exists())
                    self.assertTrue(result["pdb_complete"])
                    self.assertFalse(result["baseline_dispatched"])
                    self.assertFalse(result["five_system_complete"])
                    self.assertEqual(finish.checked(result["supervisor_snapshot"]), finish.read(state))
                else:
                    with self.assertRaises(AssertionError):
                        finish.main()
                    self.assertFalse(stop.exists())
                    self.assertFalse(out.exists())

    def test_finish_only_after_all_three_pdb_boundaries(self):
        self.run_fixture(True)

    def test_unfinished_pdb_group_cannot_be_marked_complete(self):
        self.run_fixture(False)


if __name__ == "__main__":
    unittest.main()
