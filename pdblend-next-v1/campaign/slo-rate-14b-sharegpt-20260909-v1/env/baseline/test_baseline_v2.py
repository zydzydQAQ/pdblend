"""CPU regression for the actual B control package and retained native bootstrap."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import prepare_v2 as b
p = b.p


class ControlRegistrationTests(unittest.TestCase):
    def bootstrap(self, directory):
        directory = Path(directory)
        registration, template = b.registered()
        configs = []
        for index, item in enumerate(template["instances"]):
            config = copy.deepcopy(item["engine_template"])
            config.pop("retained_weights", None)
            config.update(id="test" + str(index), tp=1, role="mixed", port=30100 + index, kv_port=31000 + 32 * index,
                peers={}, runtime_dir=str(directory / "native"), weight_cache_root=str(directory / "weights"), initial_generation=0)
            path = directory / ("engine" + str(index) + ".json")
            p.save(path, config)
            configs.append(dict(tp=1, gpus=[index], native_kind="legacy_sync_put", scheduler_cache_observed=False,
                container=dict(image=template["image"], id="retained" + str(index)), engine_config=str(path)))
        value = dict(model="14b", hostname=b.s.HOSTS["A"], instances=configs,
            fresh_node_native_identity=True, old_node_qualification_inherited=False, output_correctness_verified=False,
            correctness_gate_required_before_performance=True, host_release="original-unadapted-host",
            qualified_profile_required=dict(path="original-pdb-profile", sha256="0" * 64),
            files={i["engine_config"]: p.sha(i["engine_config"]) for i in configs})
        path = directory / "bootstrap.json"; p.save(path, value)
        return p.ref(path), registration, template

    def repair(self, directory):
        path = Path(directory) / "repair.json"
        p.save(path, dict(node="A", implementation_unchanged=True,
            reference_implementation="B actual pipeline-009 frozen baselines", failed_setup_status=None))
        return p.ref(path)

    def test_registered_cold_clock_signature_accepts_gate_keyword(self):
        registration, _ = b.registered()
        proof = b.cold_compatibility(registration)
        self.assertTrue(proof["passed"])
        self.assertIn("max_frequency=2520", proof["signature"])
        self.assertFalse(proof["gpu_constructed"])

    def test_prior_host_reproduces_original_API_failure_without_gpu(self):
        old_host = p.read(b.s.HERE / "original-B-template.json")["host_release"]
        code = "import sys,inspect;sys.path[:0]=sys.argv[1:];from ecopadg.serving.backend import ClockOwner;inspect.signature(ClockOwner).bind(object(),tuple(range(8)),max_frequency=2100)"
        result = subprocess.run([sys.executable,"-I","-B","-c",code,old_host + "/src",old_host,"/root/workspace/pdblend/.runtime-deps"],capture_output=True,text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected keyword argument 'max_frequency'", result.stderr)

    def test_bootstrap_adapter_preserves_all_native_identity_and_config_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, registration, template = self.bootstrap(tmp)
            source = p.checked(reference)
            repaired = b.adapt_bootstrap(reference,"A",tmp,registration,template,self.repair(tmp))
            actual = p.checked(repaired)
            self.assertEqual(actual["instances"], source["instances"])
            self.assertEqual(actual["host_release"], registration["host_release"])
            self.assertEqual(actual["qualified_profile_required"], registration["profile"])
            self.assertTrue(all(actual["files"][k] == v for k,v in source["files"].items()))
            self.assertEqual(b.verify_control_bootstrap(repaired)["source_bootstrap"], reference)

    def test_changed_work_limit_or_native_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, registration, template = self.bootstrap(tmp)
            source = p.checked(reference)
            source["instances"][0]["tp"] = 2
            with self.assertRaisesRegex(ValueError,"native layout"):
                b.native_template(source,template)
            source = p.checked(reference)
            path = source["instances"][0]["engine_config"]
            config = p.read(path); config["max_num_seqs"] = 16; p.save(path,config)
            source["files"][path] = p.sha(path)
            with self.assertRaisesRegex(ValueError,"startup config differs"):
                b.native_template(source,template)

    def test_registration_cannot_be_claimed_with_another_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, registration, template = self.bootstrap(tmp)
            repaired = b.adapt_bootstrap(reference,"A",tmp,registration,template,self.repair(tmp))
            value = p.checked(repaired); value["qualified_profile_required"] = p.checked(reference)["qualified_profile_required"]
            p.save(Path(tmp)/"tampered.json",value)
            with self.assertRaisesRegex(ValueError,"source/profile"):
                b.verify_control_bootstrap(p.ref(Path(tmp)/"tampered.json"))

    def test_descriptors_separate_A_retained_and_C_fresh_deployment(self):
        a,c=b.descriptor("A"),b.descriptor("C")
        self.assertIn("--bootstrap",a["argv"])
        self.assertIn("--prior-attempt",a["argv"])
        self.assertNotIn("--bootstrap",c["argv"])
        self.assertIn("baseline-qualification-002/ready.json",a["handoffs_path"])
        self.assertIn("baseline-002/ready.json",c["handoffs_path"])
        self.assertEqual(a["runtime_pythonpath"],c["runtime_pythonpath"])
        self.assertIn("platform-domain/runtime-baseline-002/src",a["runtime_pythonpath"][0])

    def test_resume_requires_the_exact_clean_terminal_setup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            bootstrap,_,_=self.bootstrap(tmp)
            dead=dict(pid=999999999,startticks="0",finished_s=1,exitcode=1)
            prior=dict(node="A",binding=bootstrap,complete=False,node_lease_held=False,finished_s=1,child=dead)
            p.save(tmp/"status.json",prior)
            p.save(tmp/"qualification/status.json",dict(complete=False,finished_s=1,node_lease_held=False,child=dead))
            native=dict(complete=True,passed=False,finished_s=1,native_cleanup_complete=True,clock_restore_complete=True,cleanup_errors=[],
                error='TypeError("ClockOwner.__init__() got an unexpected keyword argument \'max_frequency\'")')
            p.save(tmp/"qualification/native/status.json",native)
            proof=b.diagnosed_prior(p.ref(tmp/"status.json"),bootstrap,"A")
            self.assertEqual(proof["failed_setup_status"],p.ref(tmp/"status.json"))
            native["clock_restore_complete"]=False
            p.save(tmp/"qualification/native/status.json",native)
            with self.assertRaisesRegex(ValueError,"diagnosed clean setup-only"):
                b.diagnosed_prior(p.ref(tmp/"status.json"),bootstrap,"A")


if __name__ == "__main__":
    unittest.main()
