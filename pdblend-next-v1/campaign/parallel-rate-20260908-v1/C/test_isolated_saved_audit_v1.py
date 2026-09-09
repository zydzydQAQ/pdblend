import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import isolated_measurement_audit_v1 as audit
import audit_dynamic_v2 as dynamic
SELFTEST = ROOT/'A/isolated-power-hardware-selftest-002'

class Transport(unittest.TestCase):
    def setUp(self):
        terminal = audit.read(SELFTEST/'isolated-observers-terminal.json')
        row = terminal['isolated_samplers'][0]
        self.spec = audit.fixed(row['spec']); self.receipt = audit.fixed(row['receipt'])
        self.raw = audit.fixed(row['raw']); self.host = audit.fixed(self.spec['host_manifest'])
    def call(self): return audit.transport(self.spec, self.receipt, self.raw, self.host)
    def test_real_403_rows_and_saved_primary_files(self):
        result = self.call(); self.assertEqual(result['samples'], 403)
        terminal = audit.read(SELFTEST/'isolated-observers-terminal.json')
        checked = audit.audit_samplers(terminal['isolated_samplers'], self.spec['host_manifest'], artifacts=terminal['artifacts'])
        matched = audit.match_power_directory(SELFTEST/'raw', checked,
                artifacts=audit.read(SELFTEST/'raw/measurement.json')['artifacts'])
        self.assertTrue(matched['all_original_rows_exact'])
    def test_power_mutation_is_rejected_by_IPC_digest(self):
        self.raw['samples'][100][1][2] += 1
        with self.assertRaisesRegex(ValueError, 'IPC digest differs'): self.call()
    def test_metadata_mutation_rejected_by_IPC_digest(self):
        self.raw['metadata'][100]['nvml_timestamp_us'][2] += 1
        with self.assertRaisesRegex(ValueError, 'IPC digest differs'): self.call()
    def test_dropped_row_rejected(self):
        self.raw['samples'].pop()
        with self.assertRaisesRegex(ValueError, 'dropped or relabeled'): self.call()
    def test_reordered_rows_rejected(self):
        self.raw['samples'][100], self.raw['samples'][101] = self.raw['samples'][101], self.raw['samples'][100]
        with self.assertRaises(ValueError): self.call()
    def test_worker_not_exited_rejected(self):
        self.receipt['child_exited'] = False
        with self.assertRaisesRegex(ValueError, 'exit cleanly'): self.call()
    def test_old_sampling_interval_rejected(self):
        self.spec['interval'] = .1
        with self.assertRaisesRegex(ValueError, '50Hz'): self.call()
    def test_missing_gpu_rejected(self):
        self.spec['gpus'].pop()
        with self.assertRaisesRegex(ValueError, 'all8'): self.call()
    def test_wrong_power_source_rejected(self):
        self.raw['power_source']['field_id'] = 155
        with self.assertRaisesRegex(ValueError, 'source changed'): self.call()
    def test_bad_csv_not_matched_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ['power.csv', 'power_metadata.jsonl', 'power_source.json']:
                (p/name).write_bytes((SELFTEST/'raw'/name).read_bytes())
            file = p/'power.csv'; text = file.read_text(); text = text.replace('34.377', '34.378', 1); file.write_text(text)
            with self.assertRaisesRegex(ValueError, 'not exactly one original'):
                audit.match_power_directory(p, dict(raw_values={'raw': self.raw}))

class Union(unittest.TestCase):
    def test_exact_union_including_shared_transition_sampler_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'evidence'; p.write_text('real')
            files = {str(p):audit.sha(p)}
            self.assertEqual(dynamic.artifact_union(files, files, files, files), files)
    def test_omitted_sampler_rejected(self):
        with self.assertRaisesRegex(ValueError, 'references differ'):
            dynamic.artifact_union({'a':'1'}, {'b':'2'}, {'a':'1'}, {})
    def test_extra_artifact_rejected(self):
        with self.assertRaisesRegex(ValueError, 'references differ'):
            dynamic.artifact_union({'a':'1'}, {}, {'a':'1','b':'2'}, {})
    def test_capacity_off_remains_empty_without_any_measurement_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root/'config.json'; config.write_text('{}')
            binding = root/'binding.json'; binding.write_text(json.dumps(dict(configs={'dataset':str(config)})))
            receipt = root/'receipt.json'; receipt.write_text('{}')
            cp = root/'checkpoint.json'; cp.write_text(json.dumps(dict(binding=str(binding), receipt=str(receipt))))
            self.assertEqual(dynamic.inspect({'dataset':'dataset'},cp), {})

class Qualification(unittest.TestCase):
    def test_old_gate_rows_and_identity_checks_whole_AST_unchanged(self):
        old = ast.parse((ROOT/'A/p8_qualification_audit_v2.py').read_text())
        new = ast.parse((ROOT/'A/p8_qualification_audit_v3.py').read_text())
        methods = {n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
        for node in old.body:
            if isinstance(node,ast.FunctionDef) and node.name != 'audit900':
                self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(methods[node.name],include_attributes=False),node.name)

if __name__ == '__main__': unittest.main()
