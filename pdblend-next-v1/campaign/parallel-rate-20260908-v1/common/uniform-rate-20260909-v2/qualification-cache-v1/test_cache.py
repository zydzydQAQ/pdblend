import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cache


class CacheTests(unittest.TestCase):
    def fixture(self, root):
        raw=root/'raw.json';raw.write_text('{"good":true}')
        binding=root/'binding.json';binding.write_text('{}')
        weight=root/'large-input.bin';weight.write_bytes(b'weight')
        q=root/'qualified.json';q.write_text(json.dumps(dict(binding=cache.ref(binding), weight=str(weight))))
        verifier=root/'original.py'
        verifier.write_text('import json,os\nfrom pathlib import Path\n'
            'def verify(reference):\n'
            ' q=json.loads(Path(reference["path"]).read_text())\n'
            ' root=Path(reference["path"]).parent\n'
            ' raw=json.loads((root/"raw.json").read_text())\n'
            ' assert os.stat(q["weight"]).st_size==6\n'
            ' return dict(passed=raw["good"],independently_recomputed=True,binding=q["binding"],qualification=reference)\n')
        result=cache.build(cache.ref(q),cache.ref(verifier),root/'cache',workspace=root)
        return result,raw,binding,weight,q,verifier

    def test_full_proof_and_actual_dependency_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,raw,binding,weight,q,verifier=self.fixture(root)
            saved=cache.checked(result['cache'])
            self.assertIn(str(raw),saved['files']);self.assertIn(str(binding),saved['files'])
            self.assertIn(str(weight),saved['stat_only_inputs'])
            proof=cache.load(result['qualification_validator'],'test_cached_wrapper').verify(result['qualification'])
            self.assertTrue(proof['passed']);self.assertTrue(proof['full_independent_audit_reused'])
            self.assertEqual(proof['binding'],cache.ref(binding))
            self.assertEqual(proof['files'][result['cache']['path']],result['cache']['sha256'])
            self.assertEqual(proof['files'][result['helper']['path']],result['helper']['sha256'])

    def test_changed_raw_rejected_even_if_length_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            result,raw,*_=self.fixture(Path(directory))
            raw.write_text('{"good":null}')
            with self.assertRaisesRegex(ValueError,'(dependency|identity) changed'):
                cache.verify_cached(result['qualification'],result['cache'])

    def test_missing_read_dependency_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result,raw,*_=self.fixture(Path(directory));raw.unlink()
            with self.assertRaises(FileNotFoundError):
                cache.verify_cached(result['qualification'],result['cache'])

    def test_other_qualification_or_host_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result,*_=self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'another qualification'):
                cache.verify_cached(dict(result['qualification'],sha256='0'*64),result['cache'])
            with patch.object(cache.socket,'gethostname',return_value='another-host'):
                with self.assertRaisesRegex(ValueError,'another host'):
                    cache.verify_cached(result['qualification'],result['cache'])

    def test_binding_and_stat_only_inputs_remain_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            result,raw,binding,weight,*_=self.fixture(Path(directory))
            weight.write_bytes(b'changed-weight')
            with self.assertRaisesRegex(ValueError,'stat-only input changed'):
                cache.verify_cached(result['qualification'],result['cache'])
        with tempfile.TemporaryDirectory() as directory:
            result,raw,binding,*_=self.fixture(Path(directory));binding.write_text('{"model":"wrong"}')
            with self.assertRaisesRegex(ValueError,'(dependency|identity) changed'):
                cache.verify_cached(result['qualification'],result['cache'])

    def test_false_original_proof_does_not_create_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);q=root/'q.json';q.write_text('{}');verifier=root/'verify.py'
            verifier.write_text('def verify(reference): return dict(passed=False,independently_recomputed=True)\n')
            with self.assertRaisesRegex(ValueError,'did not pass'):
                cache.build(cache.ref(q),cache.ref(verifier),root/'cache',workspace=root)
            self.assertFalse((root/'cache').exists())


    def test_same_bytes_replaced_inode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result,raw,*_=self.fixture(Path(directory))
            replacement=raw.with_suffix('.replacement');replacement.write_bytes(raw.read_bytes())
            replacement.replace(raw)
            with self.assertRaisesRegex(ValueError,'identity changed'):
                cache.verify_cached(result['qualification'],result['cache'])

    def optional_fixture(self, root, statement):
        binding=root/'binding.json';binding.write_text('{}')
        q=root/'q.json';q.write_text(json.dumps(dict(binding=cache.ref(binding))))
        verifier=root/'verify.py'
        verifier.write_text('import json,os\nfrom pathlib import Path\n'
            'def verify(reference):\n'
            ' root=Path(reference["path"]).parent\n'
            + statement + '\n'
            ' return dict(passed=True,independently_recomputed=True,binding=json.loads(Path(reference["path"]).read_text())["binding"])\n')
        return cache.build(cache.ref(q),cache.ref(verifier),root/'cache',workspace=root)

    def test_absent_dependency_appears_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=self.optional_fixture(root,' assert not (root/"optional.json").exists()')
            (root/'optional.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'previously absent input appeared'):
                cache.verify_cached(result['qualification'],result['cache'])

    def test_directory_addition_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'inputs').mkdir()
            result=self.optional_fixture(root,' assert os.listdir(root/"inputs")==[]')
            (root/'inputs'/'new').write_text('x')
            with self.assertRaisesRegex(ValueError,'directory membership changed'):
                cache.verify_cached(result['qualification'],result['cache'])

    def test_fresh_worker_ignores_parent_preloaded_module(self):
        import sys, types
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fake=types.ModuleType('poisoned_workspace')
            fake.__file__=str(root/'poison.py')
            with patch.dict(sys.modules,{'poisoned_workspace':fake}):
                result,*_=self.fixture(root)
                self.assertTrue(cache.verify_cached(result['qualification'],result['cache'])['passed'])
                with self.assertRaisesRegex(ValueError,'workspace module preloaded'):
                    cache.build_worker(result['qualification'],result['qualification_validator'],root/'other',workspace=root)

    def test_unsupported_external_channels_fail_closed(self):
        for statement in [
            ' import subprocess; subprocess.run(["true"],check=True)',
            ' os.system("true")',
            ' import socket; socket.socket()',
            ' import ctypes; ctypes.CDLL(None)',
        ]:
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                with self.assertRaisesRegex(ValueError,'unsupported qualification input channel'):
                    self.optional_fixture(root,statement)
                self.assertFalse((root/'cache').exists())


    def test_read_write_open_cannot_hide_read_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'mutable.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'must not write files'):
                self.optional_fixture(root,' with open(root/"mutable.json","r+") as f: assert f.read()=="{}"')
            self.assertFalse((root/'cache').exists())

    def test_late_same_byte_replacement_hits_final_metadata_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            result,raw,binding,*_=self.fixture(Path(directory))
            original_sha=cache.sha
            files=cache.checked(result['cache'])['files']
            last=next(reversed(files))
            self.assertNotEqual(last,str(binding))
            def replace_earlier_after_hash(path):
                digest=original_sha(path)
                if str(path)==last:
                    replacement=binding.with_suffix('.replacement')
                    replacement.write_bytes(binding.read_bytes());replacement.replace(binding)
                return digest
            with patch.object(cache,'sha',side_effect=replace_earlier_after_hash):
                with self.assertRaisesRegex(ValueError,'final metadata barrier changed'):
                    cache.verify_cached(result['qualification'],result['cache'])


    def test_external_regular_file_is_a_dependency(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root=Path(directory);raw=Path(external)/'raw.txt';raw.write_text('external')
            result=self.optional_fixture(root,' assert Path('+repr(str(raw))+').read_text()=="external"')
            self.assertIn(str(raw),cache.checked(result['cache'])['files'])
            raw.write_text('modified')
            with self.assertRaisesRegex(ValueError,'changed'):
                cache.verify_cached(result['qualification'],result['cache'])


if __name__=='__main__':unittest.main()
