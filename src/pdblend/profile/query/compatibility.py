"""Check new query numerics against checksum-verified frozen implementations.

Only the selected, pure scalar functions are compiled from the frozen source;
the historical collection module is never imported or executed.
"""
import ast
import hashlib
import math
from .power_table import PowerCoverageError
from .decode import predict as decode_predict
from .index import finite


def functions(path, names, constants=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    # New frozen layouts keep the regime predicate beside the scalar table.
    # Compile the frozen helper itself, not a helper imported from live code.
    if 'context_bounds' in names and any(isinstance(n,ast.FunctionDef) and n.name == '_batch_family' for n in tree.body):
        names = (*names,'_batch_family')
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError('frozen numerical reference functions missing')
    literals = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name):
            try:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError,TypeError):
                pass
    namespace = dict(literals,math=math,PowerCoverageError=PowerCoverageError,**(constants or {}))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def numerical_sources(frozen_root, manifest, names):
    """Resolve old or reorganized *frozen* implementations, never live shims.

    A compatibility entrance alone cannot establish numerical provenance. The
    selected file must contain the implementation and have its own manifest hash.
    Historical manifests/hashes are read unchanged.
    """
    moved = {'model.py':'query/model.py', 'power_table.py':'query/power_table.py',
             'decode_fit.py':'query/decode.py', 'timing_calibration.py':'query/timing.py',
             'long_context_followup.py':'query/long_context.py'}
    required = {'model.py':{'PerfModel'}, 'power_table.py':{'context_bounds','predict'},
                'decode_fit.py':{'predict'}, 'timing_calibration.py':{'weight','TimingOverlay'},
                'long_context_followup.py':{'predict'}}
    result = {}
    for name in names:
        for relative in (name, moved[name]):
            path = frozen_root / relative
            key = 'pdblend/profile/' + relative
            if not path.is_file() or key not in manifest:
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
                raise ValueError('frozen numerical implementation checksum mismatch: '+relative)
            definitions = {n.name for n in ast.parse(path.read_text()).body
                           if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            if required[name] <= definitions:
                result[name] = path
                break
        else:
            raise ValueError('frozen numerical implementation unavailable (alias is insufficient): '+name)
    return result


def _model_methods(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PerfModel')
    names = {'prefill_seconds', 'prefill_power_w', 'prefill_marginal_seconds',
             'static_power_w', 'wake_seconds', 'transfer_seconds'}
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError('frozen model numerical methods missing')
    saturation = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'PREFILL_SATURATION' for t in n.targets)
                      and isinstance(n.value, ast.Constant))
    namespace = dict(PREFILL_SATURATION=saturation, Optional=__import__('typing').Optional, finite=finite)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def check(base, frozen_root, long_candidate=None, long_core=None, *, sources=None):
    source = lambda name: sources[name] if sources is not None else frozen_root / name
    power = functions(source('power_table.py'), ('context_bounds', 'predict'))['predict']
    decode = functions(source('decode_fit.py'), ('predict',))['predict']
    model = _model_methods(source('model.py'))
    count = 0

    def equal(left, right):
        nonlocal count
        if not math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-14):
            raise ValueError('consumer numerical compatibility check failed')
        count += 1

    for frequency in base.freqs:
        low, high = base.bounded_coverage.get('prefill_tokens', (1, 7168))
        for tokens in (low, high, (low+high)/2, min(max(1024,low),high)):
            for name in ('prefill_seconds', 'prefill_power_w', 'prefill_marginal_seconds'):
                equal(getattr(base, name)(tokens, frequency), model[name](base, tokens, frequency))
        equal(base.static_power_w('active_idle', frequency), model['static_power_w'](base, 'active_idle', frequency))
    for state in base.static:
        equal(base.wake_seconds(state), model['wake_seconds'](base, state))
    for tokens in (0, 1, 512, 8192):
        equal(base.transfer_seconds(tokens), model['transfer_seconds'](base, tokens))

    for frequency, spec in base.decode_power_overrides.items():
        batches = sorted({node['batch'] for node in spec['nodes']})
        queries = batches + [(a + b) / 2 for a, b in zip(batches, batches[1:]) if a >= 4]
        for batch in queries:
            low, high = base.decode_power_overrides.at(frequency).context_bounds(batch)
            contexts = {low, high, (low + high) / 2}
            for node in spec['nodes']:
                contexts.update(value for value in (node['context_min'], node['context_max']) if low <= value <= high)
            coordinates = sorted(contexts)
            contexts.update((a + b) / 2 for a, b in zip(coordinates, coordinates[1:]))
            for context in contexts:
                equal(base.decode_power_w(batch, frequency, ctx=context), power(spec, batch, context))
    for frequency, spec in base.decode_overrides.items():
        domain = spec['domain']
        for batch in (*domain['batch'], sum(domain['batch']) / 2, 4.0, 4.5):
            for context in (*domain['context'], sum(domain['context']) / 2):
                equal(decode_predict(spec, batch, context), decode(spec, batch, context))
    if long_candidate is not None:
        long = functions(source('long_context_followup.py'), ('predict',),
                         dict(KIND=long_candidate['kind']))['predict']
        for key, nodes in long_candidate['nodes'].items():
            frequency, batch = map(int, key.split('/'))
            contexts = [node['context'] for node in nodes]
            contexts += [(a['context'] + b['context']) / 2 for a, b in zip(nodes, nodes[1:])]
            for context in contexts:
                for metric in ('step_seconds', 'power_w'):
                    equal(long_core._long.predict(metric, frequency, batch, context),
                          long(long_candidate, metric, frequency, batch, context))
    if sources is not None and 'timing_calibration.py' in sources:
        weight = functions(source('timing_calibration.py'), ('weight',))['weight']
        for frequency, residual in long_core.candidate['residual_seconds'].items():
            frequency = int(frequency)
            domain = base.decode_overrides[frequency]['domain']
            context = sum(domain['context'])/2
            for batch in (16, 24, 32, 48, 64):
                if base.decode_supported(batch, context, frequency):
                    equal(long_core.step_seconds(batch, context, frequency),
                          base.step_seconds(batch, context, frequency)+weight(batch)*residual)
    return dict(passed=True, reference='checksum_bound_frozen_numerical_source', scalar_checks=count,
                relative_tolerance=1e-13, absolute_tolerance=1e-14)
