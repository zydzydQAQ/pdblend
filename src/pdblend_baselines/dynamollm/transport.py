"""HTTP/SSE transport for the independent Dynamo controller and V1 bridge."""
from __future__ import annotations

import json
import codecs
import time
import aiohttp


def validate_rank_ack(value, spec, *, transaction_id=None):
    ranks = value.get('ranks', [])
    if (len(ranks) != spec['tp'] or {row.get('rank') for row in ranks} != set(range(spec['tp']))
            or any(row.get('ok') is not True or row.get('generation') != spec.get('generation', 0)
                   for row in ranks)):
        raise RuntimeError('incomplete Dynamo rank/generation ACK')
    if transaction_id and any(row.get('transaction_id') != transaction_id for row in ranks):
        raise RuntimeError('Dynamo rank ACK transaction differs')
    return value


class V1Transport:
    def __init__(self, instances, clock, journal):
        self.instances = {row.get('instance_id', row.get('id')): dict(row) for row in instances}
        self.set_clock, self.journal = clock, journal
        self.session = None
        self.dynamo_topology = None

    async def start(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180, sock_connect=5))

    async def json(self, iid, path, payload=None, *, method='POST'):
        spec = self.instances[iid]
        body = dict(payload or {})
        body.setdefault('expected_generation', spec.get('generation', 0))
        try:
            async with self.session.request(method, spec['url'] + path,
                                             json=body if method != 'GET' else None) as response:
                response.raise_for_status()
                value = await response.json()
        except aiohttp.ClientError as exc:
            raise RuntimeError('Dynamo native HTTP failed: ' + iid + path) from exc
        if not isinstance(value, dict) or value.get('error'):
            raise RuntimeError('Dynamo native endpoint error: ' + repr(value))
        if path.rsplit('/', 1)[-1] in ('describe', 'open', 'transfer', 'close', 'drain'):
            validate_rank_ack(value, spec, transaction_id=body.get('transaction_id'))
        self.journal('dynamo_native_receipt', instance_id=iid, path=path,
                     transaction_id=body.get('transaction_id'), response=value)
        return value

    async def state(self, iid):
        native = await self.json(iid, '/baseline/state', method='GET')
        result = dict(native)
        # Translate the neutral V1 schema without inventing absent proof.
        # Preserve the native object in every state receipt for independent audit.
        result['native_raw'] = native
        for key in ('running', 'waiting'):
            if isinstance(native.get(key), list):
                result[key] = len(native[key])
        if 'active' not in native and isinstance(native.get('all_queue'), list):
            result['active'] = len(native['all_queue'])
        for old, new in (('timestamp', 'native_at_s'), ('evidence_complete', 'native_evidence_complete'),
                         ('transport_healthy', 'healthy')):
            if old not in native and new in native:
                result[old] = native[new]
        if (result.get('generation') != self.instances[iid].get('generation', 0)
                or result.get('evidence_complete') is not True or result.get('transport_healthy') is not True):
            raise RuntimeError('Dynamo native state identity/evidence missing')
        stamp = result.get('timestamp')
        if type(stamp) not in (int, float) or not 0 <= time.time() - stamp <= 1:
            raise RuntimeError('stale Dynamo native snapshot')
        return result

    async def clock(self, gpus, frequency):
        receipt = self.set_clock(gpus, frequency)
        self.journal('dynamo_clock', **receipt)
        return receipt

    async def cancel(self, iid, request_id):
        return await self.json(iid, '/baseline/cancel', {'request_id': request_id})

    async def stream(self, iid, payload):
        body = dict(payload, model='m', stream=True)
        buffer = ''
        decoder = codecs.getincrementaldecoder('utf-8')()
        async with self.session.post(self.instances[iid]['url'] + '/baseline/generate', json=body) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_any():
                buffer += decoder.decode(chunk)
                buffer = buffer.replace('\r\n', '\n')
                while '\n\n' in buffer:
                    block, buffer = buffer.split('\n\n', 1)
                    for line in block.splitlines():
                        if not line.startswith('data:'):
                            continue
                        raw = line[5:].strip()
                        if raw == '[DONE]':
                            return
                        event = json.loads(raw)
                        # Native token_ids or cumulative token_index are required
                        # by emergency control; text chunk counts are never tokens.
                        ids = event.get('token_ids')
                        for choice in event.get('choices', []):
                            if ids is None and choice.get('token_ids') is not None:
                                ids = choice['token_ids']
                        if ids is not None:
                            event['token_ids'] = ids
                        if ids is None and 'token_index' not in event and any(
                                choice.get('text') for choice in event.get('choices', [])):
                            raise RuntimeError('native SSE token accounting unavailable')
                        yield event

    async def close(self):
        if self.session is not None:
            await self.session.close()
