"""Read-only preservation of actual scheduler events for new ablation runs.

This process never sends engine controls or touches clocks. Each stream is
bound to the newly started process and original engine configuration, and only
complete JSON lines are committed. Missing data remains explicitly missing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DEADLINE = 1788868800.0


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_streams(attempt):
    binding_path = attempt / 'restoration/binding.json'
    b = read(binding_path)
    containers = read(attempt / 'restoration/containers.after.json')
    by_id = {c['Id']: c for c in containers}
    streams = []
    for i in b['instances']:
        container = by_id[i['container']['id']]
        assert container['State']['StartedAt'] == i['container']['StartedAt']
        argv = container['Args']
        config_path = Path(argv[argv.index('--config') + 1])
        assert b['files'][str(config_path)] == sha(config_path)
        cfg = read(config_path)
        assert cfg['id'] == i['id'] and cfg['tp'] == i['tp']
        streams.append(dict(instance_id=i['id'], container_id=container['Id'],
            started_at=container['State']['StartedAt'], binding=str(binding_path),
            binding_sha256=sha(binding_path), config=str(config_path),
            config_sha256=sha(config_path),
            source=str(Path(cfg['runtime_dir']) / (i['id'] + '.control.events.jsonl')),
            destination=str(attempt / 'telemetry' / (i['id'] + '.events.jsonl')),
            offset=0, lines=0, available=False, complete=False,
            source_may_include_historical_prefix=True,
            attribution_rule='new process identity plus actual per-cell measurement timestamps'))
    return streams


def copy_complete_lines(stream, max_bytes=8 * 1024**2):
    source, target = Path(stream['source']), Path(stream['destination'])
    if not source.is_file():
        return False
    stat = source.stat()
    identity = [stat.st_dev, stat.st_ino]
    if stream.get('source_identity') not in (None, identity):
        raise RuntimeError('owner event file replaced during bound observation')
    if stat.st_size < stream['offset']:
        raise RuntimeError('owner event stream was truncated')
    stream['source_identity'] = identity
    stream['available'] = True
    with source.open('rb') as f:
        f.seek(stream['offset'])
        data = f.read(max_bytes)
    size = data.rfind(b'\n') + 1
    if not size:
        if len(data) == max_bytes:
            raise RuntimeError('event line exceeds bounded observer buffer')
        return False
    committed = data[:size]
    rows = [json.loads(line) for line in committed.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError('scheduler event is not an object')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('ab') as f:
        f.write(committed)
    stream['offset'] += size
    stream['lines'] += len(rows)
    stream['observed_source_bytes'] = stat.st_size
    return True


def observe(model, root=ROOT, poll_s=2):
    watcher = root / 'watchers' / model / 'observer-001.json'
    attempt = root / 'attempts' / (model + '-001')
    status_path = root / 'watchers' / model / 'telemetry-001.json'
    assert not status_path.exists(), 'observer already exists; no silent replay'
    state = dict(model=model, pid=os.getpid(), started_s=time.time(),
        phase='waiting_for_fresh_binding', hardware_actions=False, streams=[])
    try:
        while time.time() < DEADLINE:
            state['checked_s'] = time.time()
            w = read(watcher) if watcher.is_file() else {}
            if not state['streams'] and (attempt / 'restoration/binding.json').is_file():
                state['streams'] = source_streams(attempt)
                if any(Path(s['destination']).exists() for s in state['streams']):
                    raise RuntimeError('new telemetry destinations already exist')
                state['phase'] = 'collecting_actual_scheduler_events'
            for stream in state['streams']:
                copy_complete_lines(stream)
            write(status_path, state)
            if w.get('finished_s'):
                state['phase'] = 'finished' if state['streams'] else 'no_measured_engine'
                break
            time.sleep(poll_s)
        else:
            state['phase'] = 'delivery_cutoff_reached'
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        for stream in state['streams']:
            dest = Path(stream['destination'])
            if dest.is_file():
                stream['sha256'] = sha(dest)
                stream['bytes'] = dest.stat().st_size
                stream['complete'] = (not state.get('error') and state.get('phase') == 'finished'
                    and stream['offset'] == stream.get('observed_source_bytes'))
        state['finished_s'] = time.time()
        write(status_path, state)
        if state['streams']:
            write(attempt / 'telemetry/manifest.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('14b', '7b', '32b'), required=True)
    observe(parser.parse_args().model)
