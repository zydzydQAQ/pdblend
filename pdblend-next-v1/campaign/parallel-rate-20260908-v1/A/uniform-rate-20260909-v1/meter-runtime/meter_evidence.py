"""Attach observer evidence to the unchanged P8 measurement computation."""
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attach(result, sampler, host, adapter, hooks):
    result['measurement_adapter'] = adapter
    directory = sampler._directory
    try:
        artifacts = hooks.completed_artifacts({directory}, host, adapter)
        result['isolated_samplers'] = hooks.sampler_references({directory})
        result['artifacts'].update(artifacts)
    except Exception as exc:
        result['measurement_valid'] = False
        result['isolated_sampler_evidence_error'] = repr(exc)
        result['isolated_samplers'] = []
        if directory is not None:
            result['artifacts'].update({str(p): sha(p) for p in directory.rglob('*') if p.is_file()})
    return result


def install(root, host, adapter, hooks_ref):
    if sha(hooks_ref['path']) != hooks_ref['sha256']:
        raise RuntimeError('measurement hooks source changed')
    spec = importlib.util.spec_from_file_location('p8_isolated_measurement_hooks', hooks_ref['path'])
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    module = hooks.install(root, host, adapter)
    observers = []
    original_init = module.IsolatedPowerSampler.__init__

    def tracked_init(sampler, *args, **kwargs):
        original_init(sampler, *args, **kwargs)
        observers.append(sampler)

    module.IsolatedPowerSampler.__init__ = tracked_init
    import capacity_backend
    original_snapshot = capacity_backend.TransitionMeter._finish_snapshot

    def snapshot(meter, finished_s):
        result = original_snapshot(meter, finished_s)
        return attach(result, meter.sampler, host, adapter, hooks)

    capacity_backend.TransitionMeter._finish_snapshot = snapshot

    def finish_observers():
        errors = []
        for sampler in observers:
            try:
                sampler.stop()
                if sampler._thread.is_alive() or sampler.error:
                    raise RuntimeError(sampler.error or 'isolated observer remained alive')
            except BaseException as exc:
                errors.append(repr(exc))
        roots = hooks.directories(root)
        try:
            artifacts = hooks.completed_artifacts(roots, host, adapter)
            references = hooks.sampler_references(roots)
        except Exception as exc:
            errors.append(repr(exc))
            artifacts = {str(p): sha(p) for d in roots for p in d.rglob('*') if p.is_file()}
            references = []
        value = dict(measurement_adapter=adapter, isolated_samplers=references,
                     artifacts=artifacts, complete=not errors, errors=errors)
        path = Path(root).parent/'isolated-observers-terminal.json'
        with path.open('x') as handle:
            json.dump(value, handle, indent=2); handle.write('\n')
        if errors:
            raise RuntimeError('isolated observer terminal: '+repr(errors))
    return finish_observers
