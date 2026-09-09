"""Diagnostic entry selection only: configuration objects are never mutated."""
import hashlib
import inspect
import time
from contextlib import contextmanager
from pathlib import Path

SCHEDULER_SHA = '572cdcb93e1af27439cf31534a7d1777eaf70f4b73432abd14f08ddfcfe9a691'
DEFAULT_SHA = '770cc45898d5f13e4372e864cb14ed0dcbd17c3a8db5a0463c37b6ede19d088a'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def verify_default(scheduler):
    from vllm.core.scheduler import Scheduler
    method = scheduler._schedule_default
    require(method.__self__ is scheduler and method.__func__ is Scheduler._schedule_default,
            'actual official bound default method required')
    source = Path(inspect.getsourcefile(Scheduler))
    require(hashlib.sha256(source.read_bytes()).hexdigest() == SCHEDULER_SHA, 'official scheduler SHA')
    # Method source includes its four-space class indentation, as frozen upstream.
    require(hashlib.sha256(inspect.getsource(method).encode()).hexdigest() == DEFAULT_SHA, 'official default method SHA')
    return dict(scheduler_sha256=SCHEDULER_SHA, default_method_sha256=DEFAULT_SHA)


@contextmanager
def select_default(engine, emit):
    saved = []
    configs = [(engine.scheduler_config, dict(vars(engine.scheduler_config)))]
    try:
        for index, scheduler in enumerate(engine.scheduler):
            proof = verify_default(scheduler)
            cfg = scheduler.scheduler_config
            require(cfg.chunked_prefill_enabled is True and engine.scheduler_config.chunked_prefill_enabled is True,
                    'original True configurations required')
            configs.append((cfg, dict(vars(cfg))))
            had = '_schedule' in vars(scheduler)
            value = vars(scheduler).get('_schedule')
            saved.append((scheduler, had, value, cfg))
            scheduler._schedule = scheduler._schedule_default
            emit(dict(kind='default_selection', owner=index, wall_s=time.time(),
                      engine_chunked=True, scheduler_chunked=True,
                      engine_scheduler_config_same=cfg is engine.scheduler_config,
                      original_instance_attribute_present=had, diagnostic_entry_override=True, **proof))
        yield
    finally:
        # Restore even if a later owner/source/emit check fails during installation.
        for scheduler, had, value, cfg in reversed(saved):
            if had: scheduler._schedule = value
            else: del scheduler._schedule
        require(all(dict(vars(cfg)) == before for cfg, before in configs), 'configuration mutated during reference')
        require(all(s.scheduler_config is cfg for s, _, _, cfg in saved), 'scheduler configuration object changed')
        if saved:
            emit(dict(kind='default_selection_restored', wall_s=time.time(),
                      owners=len(saved), config_objects_unchanged=True,
                      engine_chunked=engine.scheduler_config.chunked_prefill_enabled))
