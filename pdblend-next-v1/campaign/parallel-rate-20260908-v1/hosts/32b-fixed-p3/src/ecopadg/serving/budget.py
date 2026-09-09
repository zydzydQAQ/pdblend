"""Auditable finite extensions of an existing experiment budget.

The first budget remains at most 24 hours. A longer effective deadline requires
an immutable user-authorization snapshot and a strictly increasing append-only
revision chain. The original start and elapsed time never move. Mutable stage
progress stays in budget.json; immutable authority stays in .budget-ledger/.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid


INITIAL_LIMIT_S = 86400


def _number(value, label, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(label + ' must be a finite positive number')
    return value


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _immutable(path, value):
    """Atomically publish a new file without ever replacing a prior revision."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _authorization(value, started, original_limit, now):
    if (not isinstance(value, dict) or not isinstance(value.get('instruction'), str)
            or not value['instruction'].strip() or not isinstance(value.get('scope'), str)
            or not value['scope'].strip() or value.get('original_started_s') != started
            or value.get('original_limit_s') != original_limit
            or value.get('original_deadline_s') != started+original_limit):
        raise ValueError('explicit user authorization must bind the original campaign and scope')
    at = _number(value.get('authorized_at_s'), 'authorization time')
    if not started <= at <= now:
        raise ValueError('authorization timestamp is outside this campaign history')
    ceiling = value.get('hard_deadline_s')
    if ceiling is not None:
        _number(ceiling, 'user hard deadline')
        if ceiling <= at:
            raise ValueError('user hard deadline must follow the authorization')
    return ceiling


def read_budget(root, *, now=None, state=None):
    """Return effective started_s/limit_s/deadline_s/remaining_s and ledger IDs.

    ``state`` only supplies current Campaign progress; the immutable ledger is
    always read again. Old 86400-second manifests cannot shrink an authorized
    extension. This function performs no writes and imports no GPU dependency.
    """
    root = Path(root).resolve(); now = time.time() if now is None else now
    _number(now, 'current time')
    value = dict(_read(root/'budget.json') if state is None else state)
    limit = _number(value.get('limit_s'), 'initial budget limit')
    started = value.get('started_s')
    if started is not None: _number(started, 'original campaign start')
    ledger = root/'.budget-ledger'; origin_path = ledger/'origin.json'
    sequence = 0; authority_hash = None; revision_hash = None; revision_artifacts = {}; hard_deadline = None
    origin_limit = limit; origin_started = started
    if origin_path.is_file():
        origin = _read(origin_path)
        origin_started = _number(origin.get('started_s'), 'ledger start')
        origin_limit = _number(origin.get('initial_limit_s'), 'ledger initial limit')
        if (origin.get('schema') != 1 or origin_limit > INITIAL_LIMIT_S or started != origin_started
                or origin.get('original_deadline_s') != origin_started+origin_limit):
            raise ValueError('budget start or immutable origin changed')
        previous_hash = _sha(origin_path); effective_limit = origin_limit
        revision_artifacts[str(origin_path)] = previous_hash
        previous_time = _number(origin.get('recorded_at_s'), 'origin record time')
        paths = sorted(p for p in ledger.glob('*.json') if p.name != 'origin.json')
        for sequence, path in enumerate(paths, 1):
            if path.name != f'{sequence:08d}.json':
                raise ValueError('budget revisions must be contiguous and append-only')
            item = _read(path)
            if (item.get('schema') != 1 or item.get('sequence') != sequence
                    or item.get('previous_sha256') != previous_hash
                    or item.get('original_started_s') != origin_started
                    or item.get('original_limit_s') != origin_limit):
                raise ValueError('budget revision chain or original identity changed')
            extended = _number(item.get('limit_s'), 'extended total budget')
            at = _number(item.get('recorded_at_s'), 'revision record time')
            if (extended <= effective_limit or not previous_time <= at <= now
                    or item.get('deadline_s') != origin_started+extended
                    or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
                raise ValueError('budget revision must be a finite, strictly increasing explicit envelope')
            authority_hash = item.get('authorization_sha256')
            if not isinstance(authority_hash, str) or re.fullmatch('[0-9a-f]{64}', authority_hash) is None:
                raise ValueError('invalid budget authorization fingerprint')
            authorization = ledger/'authorizations'/(authority_hash+'.json')
            if not authorization.is_file() or _sha(authorization) != authority_hash:
                raise ValueError('immutable budget authorization missing or changed')
            ceiling = _authorization(_read(authorization), origin_started, origin_limit, at)
            if ceiling is not None:
                hard_deadline = ceiling if hard_deadline is None else min(hard_deadline, ceiling)
            if hard_deadline is not None and origin_started+extended > hard_deadline:
                raise ValueError('budget revision exceeds the recorded user hard deadline')
            revision_artifacts[str(authorization)] = authority_hash
            effective_limit = extended; previous_time = at
            revision_hash = previous_hash = _sha(path)
            revision_artifacts[str(path)] = revision_hash
        if limit > effective_limit:
            raise ValueError('mutable budget exceeds its authorized revision')
        limit = effective_limit
    elif limit > INITIAL_LIMIT_S:
        raise ValueError('budget above 24 hours requires explicit authorized revisions')
    elapsed = max(0, now-started) if started is not None else 0
    deadline = started+limit if started is not None else None
    value.update(started_s=started, limit_s=limit, elapsed_s=elapsed,
        remaining_s=max(0, limit-elapsed), deadline_s=deadline,
        original_started_s=origin_started, original_limit_s=origin_limit,
        original_deadline_s=origin_started+origin_limit if origin_started is not None else None,
        revision_seq=sequence, authorization_sha256=authority_hash, revision_sha256=revision_hash,
        revision_artifacts=revision_artifacts, hard_deadline_s=hard_deadline)
    return value


@contextmanager
def _node_lease(root):
    path = Path(root).parent/'node-experiment.lock'; path.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.get('PDBLEND_NODE_LOCK_FD')
    if inherited is not None:
        descriptor = os.dup(int(inherited))
        try:
            actual, expected = os.fstat(descriptor), path.stat()
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError('inherited descriptor is not the original node lease')
            handle = os.fdopen(descriptor, 'a'); descriptor = None
        finally:
            if descriptor is not None: os.close(descriptor)
    else:
        handle = path.open('a')
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def append_extension(root, *, authorization_path, limit_s, reason, now=None):
    """Append one finite authorized envelope at a safe node-lease boundary.

    Reuse is allowed only within the explicitly recorded authorization scope.
    This API does not infer permission, reset elapsed time or alter stage limits.
    """
    root = Path(root).resolve(); now = time.time() if now is None else now
    _number(limit_s, 'new total budget'); _number(now, 'revision time')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('explicit extension reason required')
    with _node_lease(root):
        current = read_budget(root, now=now)
        if current['started_s'] is None:
            raise ValueError('extensions require an already-started original campaign')
        if limit_s <= current['limit_s'] or current['started_s']+limit_s <= now+60:
            raise ValueError('extension must increase the envelope and leave future execution time')
        authority_bytes = Path(authorization_path).read_bytes()
        authority = json.loads(authority_bytes)
        new_ceiling = _authorization(authority, current['original_started_s'], current['original_limit_s'], now)
        ceilings = [v for v in (new_ceiling,current.get('hard_deadline_s')) if v is not None]
        if ceilings and current['started_s']+limit_s > min(ceilings):
            raise ValueError('extension exceeds the recorded user hard deadline')
        ledger = root/'.budget-ledger'; ledger.mkdir(exist_ok=True)
        origin_path = ledger/'origin.json'
        if not origin_path.exists():
            _immutable(origin_path, dict(schema=1, started_s=current['started_s'],
                initial_limit_s=current['limit_s'], original_deadline_s=current['deadline_s'],
                recorded_at_s=now, original_state_sha256=_sha(root/'budget.json')))
        # Snapshot exact authorization bytes, preserving its source fingerprint.
        authority_hash = hashlib.sha256(authority_bytes).hexdigest()
        destination = ledger/'authorizations'/(authority_hash+'.json')
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            temporary = destination.with_name('.'+authority_hash+'.'+uuid.uuid4().hex)
            try:
                with temporary.open('xb') as stream:
                    stream.write(authority_bytes); stream.flush(); os.fsync(stream.fileno())
                os.link(temporary, destination)
            finally: temporary.unlink(missing_ok=True)
        if _sha(destination) != authority_hash:
            raise ValueError('authorization snapshot differs')
        seq = current['revision_seq']+1
        previous = ledger/f'{seq-1:08d}.json' if seq > 1 else origin_path
        revision = dict(schema=1, sequence=seq, previous_sha256=_sha(previous),
            original_started_s=current['original_started_s'], original_limit_s=current['original_limit_s'],
            recorded_at_s=now, limit_s=limit_s, deadline_s=current['started_s']+limit_s,
            reason=reason, authorization_sha256=authority_hash,
            authorization_source_path=str(Path(authorization_path).resolve()))
        _immutable(ledger/f'{seq:08d}.json', revision)
        return read_budget(root, now=now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('show', 'extend'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--authorization', type=Path)
    envelope = parser.add_mutually_exclusive_group()
    envelope.add_argument('--total-hours', type=float)
    envelope.add_argument('--deadline-s', type=float)
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.action == 'extend':
        if args.authorization is None or (args.total_hours is None and args.deadline_s is None) or args.reason is None:
            parser.error('extend requires explicit authorization, finite envelope and reason')
        limit = (args.total_hours*3600 if args.total_hours is not None
                 else args.deadline_s-read_budget(args.root)['started_s'])
        value = append_extension(args.root, authorization_path=args.authorization,
                                 limit_s=limit, reason=args.reason)
    else: value = read_budget(args.root)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__': main()
