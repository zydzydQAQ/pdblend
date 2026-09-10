"""Read append-only native KV spans without modifying the serving engine."""
import hashlib
import json
from pathlib import Path

from .artifacts import write_json
from .statistics import quantiles


def begin_capture(config):
    paths = config.get('native_kv_logs', {})
    selected = {}
    for instance in config['instances']:
        name = instance['id']
        if name not in paths:
            raise ValueError('native KV log path missing for '+name)
        path = Path(paths[name])
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError('explicit local native KV log path required for '+name)
        stat = path.stat() if path.exists() else None
        selected[name] = dict(path=str(path), offset=stat.st_size if stat else 0,
                              inode=stat.st_ino if stat else None)
    return selected


def finish_capture(cursors, out):
    rows, evidence = [], {}
    for name, cursor in cursors.items():
        path = Path(cursor['path'])
        stat = path.stat() if path.exists() else None
        if cursor['inode'] is not None and (stat is None or stat.st_ino != cursor['inode']):
            raise ValueError('native KV log replaced during measurement: '+name)
        if stat and stat.st_size < cursor['offset']:
            raise ValueError('native KV log truncated during measurement: '+name)
        data = b''
        if stat:
            with path.open('rb') as handle:
                handle.seek(cursor['offset'])
                data = handle.read()
        if data and not data.endswith(b'\n'):
            raise ValueError('native KV log ends with a partial event: '+name)
        for line in data.splitlines():
            row = json.loads(line)
            if row.get('engine_id') != name or row.get('rank') != 0:
                raise ValueError('native KV event identity differs from TP1 allocation')
            rows.append(dict(row, source_log=cursor['path']))
        evidence[name] = dict(cursor, end_offset=cursor['offset']+len(data),
                              slice_sha256=hashlib.sha256(data).hexdigest())
    target = Path(out)/'native-kv.jsonl'
    with target.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row)+'\n')
    write_json(Path(out)/'native-kv-provenance.json', dict(
        source='native connector wall-clock spans; includes blocking, transport and import',
        files=evidence, copied_events=len(rows),
        output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return rows


def summarize_kv(rows, controls, start, end):
    """Match native nonces to admitted PD requests in the primary window."""
    pd = {e['request_id'] for e in controls if e.get('kind')=='admission' and e.get('request_id')
          and start <= e.get('at_s', -1) <= end
          and any(r.get('prefill_id') != r.get('decode_id')
                  for r in e.get('plan', {}).get('routes', []))}
    spans, send, receive = {}, [], []
    for row in rows:
        ids = []
        for request in row.get('request_ids', []):
            parts = request.split(':')
            if len(parts)==5 and parts[0]=='pdb' and parts[1] in pd:
                ids.append(parts[1])
        if not ids:
            continue
        began, finished = row['started_s'], row['finished_s']
        if not start <= began <= finished <= end:
            continue
        direction = row['direction']
        if direction not in ('send', 'recv', 'receive'):
            continue
        target = send if direction=='send' else receive
        target.append(finished-began)
        for request in ids:
            spans.setdefault(request, {}).setdefault('send' if direction=='send' else 'receive', []).append((began,finished))
    paired = []
    for directions in spans.values():
        if set(directions)=={'send','receive'}:
            values = directions['send']+directions['receive']
            paired.append(max(b for a,b in values)-min(a for a,b in values))
    return dict(kv_transfer_s=quantiles(paired), kv_send_s=quantiles(send),
        kv_receive_s=quantiles(receive), kv_transfer_count=len(paired),
        kv_pd_request_count=len(pd), kv_unpaired_pd_requests=len(pd)-len(paired),
        kv_timing_definition='Native send/receive enclosing span, including blocking and import; not pure network latency')
