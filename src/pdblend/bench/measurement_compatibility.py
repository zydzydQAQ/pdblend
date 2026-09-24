"""Explicit, narrowly reviewed compatibility for frequency-only telemetry repairs.

This is not a source identity override or a qualification waiver. Each frozen
source is checked again when its compatibility manifest is loaded.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .comparison_campaign import binding, load_bound
from .resident_session import digest, file_sha

SCHEMA = 'pdblend-measurement-compatibility/v1'
POWER = 'pdblend/measure/power.py'
BACKENDS = 'pdblend/measure/backends.py'
WHOLE_FILES = ('pdblend/bench/comparison_metering.py',
               'pdblend/bench/comparison_metrics.py', 'pdblend/bench/client.py')
# Reviewed implementations, not an arbitrary function-name exclusion. Changes
# to either body require another review before this compatibility rule expands.
REVIEWED_TELEMETRY = {
    'capture_frequency': '89730f5b7fb98dbeb5c8693742c5fd66ad19e340eda8dbab580df86865c2754a',
    'clock_diagnostics': '11715ff04b16905f9a3030c12a1cf9a272d366b394257cc3197b5e277750e7c7',
}
CORE = {
    POWER: ('instant_power_verified', 'trapezoid_energy', 'trapezoid_mean_power',
            'PowerSampler._read', 'PowerSampler.mean_power_w', 'PowerSampler.total_energy_j'),
    BACKENDS: ('PynvmlBackend.power_mode', 'PynvmlBackend.power_source',
               'PynvmlBackend.power_reading', 'PynvmlBackend.power_w'),
}


def _ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _definition(tree, name):
    for part in name.split('.'):
        tree = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if tree is None:
            raise ValueError('missing energy implementation: ' + name)
    return tree


def _normalized(tree, filename):
    """Remove only precisely reviewed telemetry additions, retaining all power code."""
    tree = deepcopy(tree)
    cls = _definition(tree, 'PowerSampler' if filename == POWER else 'PynvmlBackend')
    optional = 'capture_frequency' if filename == POWER else 'clock_diagnostics'
    for node in list(cls.body):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == optional:
            if _ast_sha(node) != REVIEWED_TELEMETRY[optional]:
                raise ValueError('unreviewed frequency telemetry body: ' + optional)
            cls.body.remove(node)
        elif filename == POWER and node.name in ('__init__', 'start'):
            retained = []
            for statement in node.body:
                target = (statement.target if isinstance(statement, ast.AnnAssign)
                          else statement.targets[0] if isinstance(statement, ast.Assign)
                          and len(statement.targets) == 1 else None)
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == 'self' and target.attr in (
                            'frequency_readings', 'frequency_errors', 'frequency_requested', '_frequency_lock')):
                    expected = ('threading.Lock()' if target.attr == '_frequency_lock'
                                else 'None' if target.attr == 'frequency_requested' else '[]')
                    if ast.dump(statement.value) != ast.dump(ast.parse(expected, mode='eval').body):
                        raise ValueError('unreviewed telemetry initialization: ' + target.attr)
                else:
                    retained.append(statement)
            node.body = retained
        elif filename == POWER and node.name == '_loop':
            old = ast.parse('self.frequency_samples.append((sample[0], [self.backend.current_freq(g) for g in self.gpus]))').body[0]
            new = ast.parse("self.capture_frequency(reason='periodic')").body[0]
            class FrequencyBlock(ast.NodeTransformer):
                def visit_If(self, block):
                    if ast.dump(block.test) == ast.dump(ast.parse('self.sample_clocks', mode='eval').body):
                        if block.orelse or len(block.body) != 1 or _ast_sha(block.body[0]) not in (_ast_sha(old), _ast_sha(new)):
                            raise ValueError('unreviewed periodic frequency block')
                        block.body = [old]
                    return self.generic_visit(block)
            FrequencyBlock().visit(node)
    return _ast_sha(tree)


def _source(ref):
    ref = binding(ref) if not isinstance(ref, dict) else dict(ref)
    manifest = load_bound(ref)
    files = manifest['files']
    if manifest.get('source_sha256') != digest(files):
        raise ValueError('frozen source identity differs from its file manifest')
    names = sorted(k for k in files if k.startswith('pdblend/measure/') or k in WHOLE_FILES)
    if not set((POWER, BACKENDS, *WHOLE_FILES)) <= set(names):
        raise ValueError('source lacks the complete energy implementation')
    root = Path(ref['path']).parent.resolve()
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha(path) != files[name]:
            raise ValueError('frozen measurement file checksum differs: ' + name)
    trees = {name: ast.parse((root / name).read_text()) for name in (POWER, BACKENDS)}
    core = {}
    for name, definitions in CORE.items():
        source = (root / name).read_text()
        for definition in definitions:
            node = _definition(trees[name], definition)
            core[name + ':' + definition] = dict(ast_sha256=_ast_sha(node),
                source_sha256=hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest())
    return dict(source_manifest=ref, source_sha256=manifest['source_sha256'],
                measurement_source_sha256=digest({k: files[k] for k in names}),
                measurement_files={k: files[k] for k in names}, energy_core=core,
                normalized_energy_ast={name: _normalized(tree, name) for name, tree in trees.items()})


def prepare_compatibility(left_source, right_source):
    """Review two arbitrary frozen revisions, refusing non-telemetry energy changes."""
    left, right = _source(left_source), _source(right_source)
    if left['measurement_files'].keys() != right['measurement_files'].keys():
        raise ValueError('measurement file inventory differs')
    changed = [k for k in left['measurement_files']
               if left['measurement_files'][k] != right['measurement_files'][k]]
    if set(changed) - {POWER, BACKENDS}:
        raise ValueError('energy implementation changed outside frequency telemetry: ' + ', '.join(changed))
    if (left['energy_core'] != right['energy_core']
            or left['normalized_energy_ast'] != right['normalized_energy_ast']):
        raise ValueError('energy core or non-frequency measurement implementation changed')
    return dict(schema=SCHEMA, compatible=True, sources=[left, right], changed_files=changed,
                change_scope=['frequency acquisition timestamps', 'frequency diagnostics',
                              'optional clock failure isolation'],
                source_identity_equal=left['source_sha256'] == right['source_sha256'],
                requirements=dict(raw_eight_gpu_meter_gate_each=True, complete_service_plus_tail_each=True,
                                  candidate_measurement_qualification=True, formal_qualification_unchanged=True))


def load_compatibility(path):
    ref = binding(path) if not isinstance(path, dict) else dict(path)
    value = load_bound(ref)
    if value.get('schema') != SCHEMA or len(value.get('sources', [])) != 2:
        raise ValueError('invalid measurement compatibility manifest')
    reviewed = prepare_compatibility(*(s['source_manifest'] for s in value['sources']))
    if value != reviewed:
        raise ValueError('compatibility manifest differs from independently recomputed review')
    return dict(reviewed, manifest_binding=ref)


def hydrate_receipt_evidence(row):
    """Recover gates only from the snapshot's hash-bound receipt and point artifact."""
    result = dict(row)
    for field in ('raw_eight_gpu_meter_qualified', 'evidence_source_manifest', 'measurement_evidence_binding'):
        result.pop(field, None)
    path, expected = row.get('receipt_path'), row.get('receipt_sha256')
    if not path or not expected:
        return result
    receipt_ref = dict(path=str(Path(path).resolve()), sha256=expected)
    receipt = load_bound(receipt_ref)
    point_path = Path(path).parent / 'point.json'
    if receipt.get('artifacts', {}).get('point.json') != file_sha(point_path):
        raise ValueError('receipt point artifact checksum differs')
    point = json.loads(point_path.read_text())
    if digest(point) != receipt.get('point_sha256') or point.get('name') != row.get('point_id'):
        raise ValueError('receipt point identity differs from cohort row')
    source_ref = point['source_manifest']
    source = load_bound(source_ref)
    if source['source_sha256'] != row.get('revision'):
        raise ValueError('cohort revision differs from receipt source')
    meter_sha = point['engine_identity']['measurement_source_sha256']
    if (row.get('comparison_identity') or {}).get('measurement_source_sha256') != meter_sha:
        raise ValueError('cohort measurement identity differs from receipt')
    native = receipt['result']
    audit = native.get('acceptance') or native.get('observation_acceptance') or {}
    gate = 'metering.raw_eight_gpu_window'
    checked = gate in audit.get('checked_gates', [])
    rejected = any(gate in audit.get(key, {}) for key in ('missing_gates', 'gate_failures', 'blocked_gates'))
    result.update(raw_eight_gpu_meter_qualified=checked and not rejected,
                  evidence_source_manifest=source_ref,
                  measurement_evidence_binding=dict(receipt=receipt_ref, acceptance_sha256=digest(audit)))
    qualification = native.get('measurement_evidence_valid')
    # Receipt evidence takes precedence over a stale derived snapshot flag.
    result.pop('measurement_qualified', None)
    result.pop('measurement_evidence_valid', None)
    if isinstance(qualification, bool):
        result['measurement_evidence_valid'] = qualification
    return result


def compatible_pair(left, right, reviews):
    """Hash differences require the exact reviewed sources and actual meter gates."""
    if not all(row.get('raw_eight_gpu_meter_qualified') is True
               and row.get('measurement_evidence_binding') for row in (left, right)):
        return None
    for review in reviews:
        sources = review['sources']
        for first, second in (sources, sources[::-1]):
            if all(row.get('evidence_source_manifest') == endpoint['source_manifest']
                   and (row.get('comparison_identity') or row).get('measurement_source_sha256')
                   == endpoint['measurement_source_sha256']
                   for row, endpoint in ((left, first), (right, second))):
                return review['manifest_binding']
    return None
