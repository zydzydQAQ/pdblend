import ast
import copy
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'dynamic-execution-until-complete-001'

def parsed(path): return ast.parse(path.read_text())
def tree(node): return ast.dump(node, include_attributes=False)
def function(module, name): return next(n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
def contains(node, value): return value in ast.unparse(node)

class SourceContract(unittest.TestCase):
    def test_ownership_and_protocol_bytes_unchanged(self):
        for name in ('dynamic_ownership.py', 'protocol.py'):
            self.assertEqual((ROOT/name).read_bytes(), (PARENT/name).read_bytes())

    def test_all_other_outer_functions_AST_identical(self):
        old = parsed(PARENT/'dynamic_measurement.py'); new = parsed(ROOT/'dynamic_measurement.py')
        for node in old.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != 'run_one':
                self.assertEqual(tree(node), tree(function(new, node.name)), node.name)

    def test_only_measurement_hooks_added_to_outer_complete_AST(self):
        old = function(parsed(PARENT/'dynamic_measurement.py'), 'run_one')
        new = function(parsed(ROOT/'dynamic_measurement.py'), 'run_one')
        new.body = [n for n in new.body if not (
            contains(n, 'sampler_hooks.') and not isinstance(n, ast.Try)
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sampler_root' for t in n.targets)
            or isinstance(n, ast.Assign) and any(contains(t, "receipt['measurement_adapter']") for t in n.targets))]
        moved = next(n for n in new.body if isinstance(n, ast.ImportFrom) and n.module == 'ecopadg.measure.power')
        new.body.remove(moved); new.body.insert(0, moved)
        job = next(n for n in new.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'job' for t in n.targets))
        job.value.keywords = [k for k in job.value.keywords if k.arg not in ('host_manifest', 'adapter_manifest')]
        body = next(n for n in new.body if isinstance(n, ast.Try))
        self.assertTrue(contains(body.body[0], 'sampler.wait_ready'))
        body.body.pop(0)
        body.finalbody = [n for n in body.finalbody if not (isinstance(n, ast.Try) and contains(n, 'sampler_hooks.completed_artifacts'))]
        self.assertEqual(tree(new), tree(old))

    def test_child_complete_AST_only_import_order_and_observer_wrapper(self):
        old = function(parsed(PARENT/'dynamic_child.py'), 'execute')
        new = function(parsed(ROOT/'dynamic_child.py'), 'execute')
        new.body = [n for n in new.body if not (
            isinstance(n, ast.Import) and any(v.name == 'sampler_hooks' for v in n.names)
            or isinstance(n, ast.ImportFrom) and n.module == 'ecopadg.serving'
            or isinstance(n, ast.Assign) and contains(n, 'sampler_hooks.install'))]
        new.body.insert(1, copy.deepcopy(old.body[1]))
        class RemovePrimaryWrapper(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'sampler_hooks.prepared_primary':
                    return ast.Call(func=ast.Name(id='run_cell', ctx=ast.Load()), args=[node.args[0]], keywords=[])
                return self.generic_visit(node)
        new = RemovePrimaryWrapper().visit(new)
        self.assertEqual(tree(new), tree(old))

if __name__ == '__main__': unittest.main()
