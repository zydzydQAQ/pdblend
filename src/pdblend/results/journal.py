"""Versioned, lossless compact journals; external engine SSE is unchanged.

One payload definition is shared by native/client observations. Cumulative
text is represented by exact prefix edits, not copied into each token row.
Readers accept historical JSONL and new streaming gzip without truncation
recovery: an incomplete gzip trailer is an invalid evidence artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = 'pdblend-journal-v1'


def artifact_path(path):
    path = Path(path)
    if path.exists():
        return path
    compressed = Path(str(path)+'.gz')
    if compressed.exists():
        return compressed
    raise FileNotFoundError(path)


@contextmanager
def open_text(path, mode='rt'):
    path = artifact_path(path) if mode.startswith('r') else Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, mode, encoding='utf-8') as handle:
        yield handle


def read_json(path):
    with open_text(path) as handle:
        return json.load(handle)


def iter_jsonl(path):
    with open_text(path) as handle:
        for index, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'journal row {index} is not an object')
                yield value


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _patch(previous, current):
    # Detokenizers may revise an incomplete Unicode suffix. Preserve those
    # revisions exactly rather than assuming cumulative text is append-only.
    if current.startswith(previous):
        return [len(previous), current[len(previous):]]
    length = 0
    for a, b in zip(previous, current):
        if a != b:
            break
        length += 1
    return [length, current[length:]]


class CompactJournal:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix != '.gz':
            raise ValueError('new compact journals require an explicit .gz suffix')
        self.raw = gzip.open(self.path, 'xt', encoding='utf-8', compresslevel=3)
        self._payloads, self._previous, self._text, self._token_refs = {}, {}, {}, {}
        self.rows = 0

    def _payload(self, payload, request_id):
        digest = hashlib.sha256(_encoded(payload).encode()).hexdigest()
        # Include request identity even if a fixture omitted it in its SSE.
        key = (request_id, digest)
        if key in self._payloads:
            return self._payloads[key], None
        event_id = f'p{len(self._payloads)}'
        self._payloads[key] = event_id
        body = dict(payload)
        patches = {}
        if isinstance(body.get('text'), str):
            patches['text'] = _patch(self._text.get((request_id, 'text'), ''), body.pop('text'))
            self._text[(request_id, 'text')] = payload['text']
        if isinstance(body.get('choices'), list):
            choices = []
            for index, choice in enumerate(body['choices']):
                choice = dict(choice)
                if isinstance(choice.get('text'), str):
                    current = choice.pop('text')
                    if current == payload.get('text'):
                        choice['_text_from_payload'] = True
                    else:
                        name = f'choice:{index}'
                        patches[name] = _patch(self._text.get((request_id, name), ''), current)
                        self._text[(request_id, name)] = current
                choices.append(choice)
            body['choices'] = choices
        definition = dict(body=body, text_patches=patches, previous=self._previous.get(request_id))
        self._previous[request_id] = event_id
        if isinstance(body.get('token_ids'), list):
            self._token_refs[(request_id, body.get('token_index'), tuple(body['token_ids']))] = event_id
        return event_id, definition

    def write(self, row):
        if self.raw is None:
            self.raw = gzip.open(self.path, 'at', encoding='utf-8', compresslevel=3)
        record = dict(row)
        if record.get('_journal_schema') is not None:
            raise ValueError('journal schema field is reserved')
        record['_journal_schema'] = SCHEMA
        if isinstance(record.get('payload'), dict):
            payload = record.pop('payload')
            request_id = record.get('request_id', payload.get('request_id'))
            event_id, definition = self._payload(payload, request_id)
            record['payload_ref'] = event_id
            if definition is not None:
                record['payload_data'] = definition
        if isinstance(record.get('token_ids'), list):
            key = (record.get('request_id'), record.get('token_index'), tuple(record['token_ids']))
            reference = self._token_refs.get(key)
            if reference is not None:
                record['token_payload_ref'] = reference
                record.pop('token_ids')
        self.raw.write(_encoded(record)+'\n')
        self.rows += 1
        return record.get('payload_ref')

    def flush(self):
        if self.raw is not None:
            self.raw.flush()

    def checkpoint(self):
        """Finish the current gzip member; reopen lazily on the next record.

        A completed stage can be read without accepting truncated evidence.
        The eventual file is a standards-compliant concatenated gzip stream.
        """
        if self.raw is not None:
            self.raw.close()
            self.raw = None

    def close(self):
        self.checkpoint()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def iter_journal(path):
    """Restore legacy-shaped rows with exact timestamps, token IDs and text.

    The reference dictionary stores only patches, never all cumulative texts.
    This keeps the reader's retained storage linear too. Repeated observations
    can require walking a request's short patch chain; this is offline replay.
    """
    definitions = {}
    for row in iter_jsonl(path):
        schema = row.pop('_journal_schema', None)
        if schema is None:
            yield row
            continue
        if schema != SCHEMA:
            raise ValueError('unsupported compact journal schema: '+str(schema))
        if 'payload_ref' in row:
            reference = row.pop('payload_ref')
            definition = row.pop('payload_data', None)
            if definition is not None:
                if reference in definitions:
                    raise ValueError('duplicate payload definition')
                definitions[reference] = definition
            if reference not in definitions:
                raise ValueError('unresolved compact payload reference')
            chain, cursor, seen = [], reference, set()
            while cursor is not None:
                if cursor in seen or cursor not in definitions:
                    raise ValueError('invalid compact payload text chain')
                seen.add(cursor)
                item = definitions[cursor]
                chain.append(item)
                cursor = item['previous']
            texts = {}
            for item in reversed(chain):
                for key, (prefix, suffix) in item['text_patches'].items():
                    prior = texts.get(key, '')
                    if type(prefix) is not int or not 0 <= prefix <= len(prior):
                        raise ValueError('invalid compact text prefix')
                    texts[key] = prior[:prefix]+suffix
            body = dict(definitions[reference]['body'])
            if 'text' in definitions[reference]['text_patches']:
                body['text'] = texts['text']
            if isinstance(body.get('choices'), list):
                choices = []
                for index, choice in enumerate(body['choices']):
                    choice = dict(choice)
                    if choice.pop('_text_from_payload', False):
                        choice['text'] = body['text']
                    elif f'choice:{index}' in definitions[reference]['text_patches']:
                        choice['text'] = texts[f'choice:{index}']
                    choices.append(choice)
                body['choices'] = choices
            row['payload'] = body
        if 'token_payload_ref' in row:
            reference = row.pop('token_payload_ref')
            if reference not in definitions:
                raise ValueError('unresolved compact token reference')
            row['token_ids'] = list(definitions[reference]['body']['token_ids'])
        yield row


def payload_receipt(events, *, journal_path, request_id):
    """Small request receipt; actual token values live only in the journal."""
    ids = [token for event in events for token in event.get('token_ids', [])]
    return dict(journal_path=str(journal_path), request_id=request_id,
                completion_tokens=len(ids), token_ids_sha256=hashlib.sha256(_encoded(ids).encode()).hexdigest(),
                event_count=len(events), terminal=bool(events and events[-1].get('finished')))


def request_events(path, request_id, kind):
    return [row['payload'] for row in iter_journal(path)
            if row.get('request_id') == request_id and row.get('kind', row.get('event')) == kind]
