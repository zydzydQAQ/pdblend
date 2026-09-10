"""CPU checks for ownership handoff and preserving qualified legacy semantics."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare as b
import sources as s
import verify as v
import contract as c
p = s.p


class BaselineHandoffTests(unittest.TestCase):
    def boundary(self, directory, *, node="A"):
        directory = Path(directory)
        refs, observations = [], []
        for repeat, q in ((1, .8), (2, .95)):
            row = c.make_row(node, ".25", "pdblend", repeat)
            observation = dict(row, measurement_valid=True, service_terminal_valid=True,
                strict_slo_recomputed=True, independently_recomputed=True,
                unknown_error_count=0, work_complete=True, slo_attainment=q)
            target = directory / ("audit" + str(repeat) + ".json")
            p.save(target, observation)
            reference = p.ref(target)
            refs.append(reference)
            observations.append(dict(observation, audit_reference=reference))
        last = dict(complete=True, finished_s=1, node_lease_held=False,
                    cleanup_complete=True, pid=999999999, startticks="0")
        p.save(directory / "last.json", last)
        p.save(directory / "binding.json", dict(model="14b", system="pdblend", hostname=s.HOSTS[node]))
        boundary = dict(campaign_id=c.CAMPAIGN, node=node, model="14b", dataset="sharegpt",
            pdb_boundary_complete=True, binding=p.ref(directory / "binding.json"),
            last_cell_status=p.ref(directory / "last.json"), observations=refs,
            observation_values=observations, pid=os.getpid(), node_lease_held=False)
        p.save(directory / "boundary.json", boundary)
        return p.ref(directory / "boundary.json")

    def test_live_supervisor_is_allowed_after_terminal_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = self.boundary(tmp)
            saved, decision = b.terminal(reference, "A")
            self.assertEqual(saved["pid"], os.getpid())
            self.assertEqual(decision["status"], "cap_confirmed")
            self.assertEqual(decision["cap_rate_rps"], .25)

    def test_live_or_unclean_cell_is_rejected(self):
        for modification in (dict(pid=os.getpid(), startticks=p.process_identity(os.getpid())["startticks"]),
                             dict(cleanup_complete=False), dict(node_lease_held=True)):
            with tempfile.TemporaryDirectory() as tmp:
                reference = self.boundary(tmp)
                saved = p.checked(reference)
                last = p.checked(saved["last_cell_status"])
                last.update(modification)
                p.save(Path(tmp) / "bad-last.json", last)
                saved["last_cell_status"] = p.ref(Path(tmp) / "bad-last.json")
                p.save(Path(tmp) / "bad-boundary.json", saved)
                with self.assertRaises(ValueError):
                    b.terminal(p.ref(Path(tmp) / "bad-boundary.json"), "A")

    def test_boundary_rejects_cross_node_or_changed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = self.boundary(tmp)
            with self.assertRaises(ValueError):
                b.terminal(reference, "C")
            saved = p.checked(reference)
            saved["observation_values"][0]["slo_attainment"] = .99
            p.save(Path(tmp) / "changed.json", saved)
            with self.assertRaises(ValueError):
                b.terminal(p.ref(Path(tmp) / "changed.json"), "A")

    def test_normal_scheduler_attempt_metadata_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = p.checked(self.boundary(tmp))
            for value in saved["observation_values"]:
                value["engineering_attempt"] = 1
            target = Path(tmp) / "scheduler-boundary.json"
            p.save(target, saved)
            self.assertEqual(b.terminal(p.ref(target), "A")[1]["status"], "cap_confirmed")

    def test_diagnosed_replacement_metadata_requires_original_invalid_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            saved = p.checked(self.boundary(tmp))
            failed = p.checked(saved["observations"][0])
            failed.update(measurement_valid=False, unknown_error_count=1)
            p.save(tmp / "failed.json", failed)
            p.save(tmp / "repair.json", {"diagnosis": "fixture engineering failure repaired"})
            failed_ref, repair_ref = p.ref(tmp / "failed.json"), p.ref(tmp / "repair.json")
            saved["observation_values"][0].update(engineering_attempt=2, repair_reference=repair_ref)
            saved["observations"].insert(0, failed_ref)
            saved["observation_values"].insert(0, dict(failed, audit_reference=failed_ref, engineering_attempt=1))
            p.save(tmp / "replacement-boundary.json", saved)
            self.assertEqual(b.terminal(p.ref(tmp / "replacement-boundary.json"), "A")[1]["status"], "cap_confirmed")
            saved["observations"].pop(0)
            saved["observation_values"].pop(0)
            p.save(tmp / "missing-original.json", saved)
            with self.assertRaisesRegex(ValueError, "missing earlier engineering attempt"):
                b.terminal(p.ref(tmp / "missing-original.json"), "A")

    def test_bad_attempt_or_repair_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = p.checked(self.boundary(tmp))
            audit_ref, value = saved["observations"][0], saved["observation_values"][0]
            p.save(Path(tmp) / "repair.json", {"diagnosis": "fixture"})
            repair = p.ref(Path(tmp) / "repair.json")
            metadata = [dict(engineering_attempt=True), dict(engineering_attempt=0),
                        dict(engineering_attempt=3), dict(engineering_attempt=2),
                        dict(engineering_attempt=1, repair_reference=repair),
                        dict(engineering_attempt=2, repair_reference=dict(repair, sha256="0" * 64)),
                        dict(engineering_attempt=1, undeclared_scheduler_fact=True)]
            for extra in metadata:
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    b.observation_with_metadata(audit_ref, dict(value, **extra))

    def test_new_spec_preserves_native_limits_and_isolates_outputs(self):
        template = p.read(s.HERE / "original-B-template.json")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            boundary = self.boundary(tmp)
            binding = p.checked(boundary)["binding"]
            spec_ref = b.make_spec(template, boundary, binding, "C", tmp / "fresh")
            spec = p.checked(spec_ref)
            self.assertEqual(spec["hostname"], s.HOSTS["C"])
            self.assertEqual(spec["node"], "C")
            self.assertEqual(len(spec["instances"]), 8)
            self.assertEqual(spec["required_predecessors"], [])
            self.assertEqual((spec["deployment_budget_s"], spec["cleanup_budget_s"]), (720, 120))
            for index, instance in enumerate(spec["instances"]):
                config = p.read(instance["config"])
                self.assertTrue(Path(instance["config"]).is_relative_to(tmp / "fresh"))
                self.assertEqual(instance["gpus"], [index])
                self.assertEqual((config["tp"], config["max_model_len"],
                    config["max_num_batched_tokens"], config["max_num_seqs"]), (1, 8192, 8192, 32))
                self.assertEqual(instance["native_kind"], "legacy_sync_put")
                self.assertFalse(instance["scheduler_cache_observed"])
                self.assertEqual(instance["image"], template["image"])
                self.assertEqual(instance["engine_entry"], template["source_entry"])

    def test_wrong_host_rejected_before_hardware_or_output_creation(self):
        with patch.object(b.socket, "gethostname", return_value="not-declared-node"):
            with self.assertRaises(ValueError):
                b.run("A", "/absent", s.NEW / "A/baseline-do-not-create")
        self.assertFalse((s.NEW / "A/baseline-do-not-create").exists())

    def test_command_descriptor_requires_terminal_and_new_output(self):
        descriptor = b.command_descriptor("C")
        self.assertEqual(descriptor["schema"], "slo-rate-baseline-command-v1")
        self.assertIn("{pdb_boundary}", descriptor["argv"])
        self.assertIn("--run", descriptor["argv"])
        self.assertEqual(descriptor["handoffs_path"], str(s.NEW / "C/baseline-001/ready.json"))
        host = Path(p.read(s.HERE / "original-B-template.json")["host_release"])
        self.assertEqual(descriptor["runtime_pythonpath"],
            [str(host / "src"), str(host), "/root/workspace/pdblend/.runtime-deps"])

    def test_raw_verifier_failure_is_never_promoted_by_hostname_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "binding.json"
            p.save(target, dict(model="14b", hostname=s.HOSTS["C"], configs={"sharegpt": "example"},
                fresh_legacy_qualification=dict(node="C", old_node_qualification_inherited=False)))
            class Original:
                def verify(self, reference):
                    if self.HOSTNAMES != {s.NATIVE_NODES[n]: h for n, h in s.HOSTS.items()}:
                        raise AssertionError("host mapping not explicit")
                    raise ValueError("original native raw mismatch")
            with patch.object(s, "checked_dependencies", return_value={"verifier": "frozen"}), \
                 patch.object(p, "load", return_value=Original()):
                with self.assertRaisesRegex(ValueError, "original native raw mismatch"):
                    v.verify(p.ref(target))


if __name__ == "__main__":
    unittest.main()
