"""Freeze identical optional capacity hooks above the three complete P4 sources."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'main-slo-improvement-v1/common'
CAPACITY = ROOT / 'A/cold-code-until-complete-001'
MODULES = ('capacity_runtime.py', 'capacity_executor.py', 'capacity_backend.py',
           'capacity_certificate.py', 'capacity_planner.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_builder():
    sys.path.insert(0, str(PREVIOUS))
    spec = importlib.util.spec_from_file_location('prior_capacity_hooks', PREVIOUS / 'build_dynamic_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DisabledCapacity(ast.NodeTransformer):
    """Specialize only the three added fields to their flag-disabled values."""
    def visit_AsyncFunctionDef(self, node):
        if node.name == 'capacity_control':
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            target = node.targets[0]
            if isinstance(target.value, ast.Name) and target.value.id == 'self' and target.attr in (
                    'capacity_service', 'capacity_task', 'capacity_enabled'):
                return None
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr == 'capacity_enabled':
            return ast.Constant(False)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) == 3:
            obj, key, default = node.args
            if isinstance(obj, ast.Name) and obj.id == 'self' and isinstance(key, ast.Constant) and key.value in (
                    'capacity_service', 'capacity_task'):
                return ast.Constant(None)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.And):
            if any(isinstance(v, ast.Constant) and not v.value for v in node.values):
                return ast.Constant(False)
            node.values = [v for v in node.values if not (isinstance(v, ast.Constant) and v.value is True)]
            if len(node.values) == 1:
                return node.values[0]
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        node.elts = [x for x in node.elts if not (isinstance(x, ast.Constant) and x.value is None)]
        return node


def default_equivalent(original, candidate):
    before = ast.dump(ast.parse(original), include_attributes=False)
    after = ast.dump(DisabledCapacity().visit(ast.parse(candidate)), include_attributes=False)
    if before != after:
        raise ValueError('capacity-disabled runtime is not the complete original P4 AST')
    return hashlib.sha256(before.encode()).hexdigest()


def build():
    builder = load_builder()
    frozen = {str(path): sha(path) for path in (
        Path(__file__), PREVIOUS / 'build_dynamic_runtime.py', PREVIOUS / 'build_runtime.py',
        *(CAPACITY / name for name in MODULES))}
    shared = None
    for model in ('7b', '14b', '32b'):
        parent = ROOT / 'hosts' / f'{model}-fixed-p4'
        out = ROOT / 'hosts' / f'{model}-capacity-p5'
        if out.exists():
            raise FileExistsError(out)
        manifest = json.loads((parent / 'manifest.json').read_text())
        runtime_name = 'src/ecopadg/serving/runtime.py'
        original = (parent / runtime_name).read_text()
        candidate = builder.patched(original)
        default_ast = default_equivalent(original, candidate)
        for name, expected in manifest['files'].items():
            if sha(parent / name) != expected:
                raise ValueError('P4 frozen source changed: ' + name)
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == runtime_name:
                dest.write_text(candidate)
            else:
                shutil.copyfile(parent / name, dest)
        for name in MODULES:
            shutil.copyfile(CAPACITY / name, out / name)
        files = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
        if shared is not None and files != shared:
            raise ValueError('model source files differ')
        shared = files
        result = dict(schema=5, model=model, implementation_series='parallel-p5', created_s=time.time(),
            files=files, frozen_references=frozen,
            parent_manifest=dict(path=str(parent / 'manifest.json'), sha256=sha(parent / 'manifest.json')),
            common_controller_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            identical_source_files_all_models=True, feature='capacity_integration_v1', default_enabled=False,
            changed_parent_files=[runtime_name], added_files=list(MODULES),
            capacity_disabled_complete_parent_ast_sha256=default_ast,
            inherited_first_admission_and_frequency_uncertainty_fixes=True,
            calibration_required_at_start=True, workload_end_restores_initial_layout=True,
            gpu_qualified=False, profiles_unchanged=True)
        (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        print(model, sha(out / 'manifest.json'), result['common_controller_sha256'])
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError('frozen builder dependency changed: ' + path)


if __name__ == '__main__':
    build()
