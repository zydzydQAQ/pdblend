import gzip
import importlib
import json
import pickle

import pytest

from pdblend.results.journal import CompactJournal, iter_journal, iter_jsonl, payload_receipt, read_json


def rows(count):
    for i in range(count):
        text = '中abc'*(i+1)
        payload = dict(request_id='r', token_ids=[i+100], token_index=i+1, text=text,
                       choices=[dict(index=0, text=text, finish_reason='length' if i==count-1 else None)],
                       at_s=10+i/10, finished=i==count-1, usage=dict(completion_tokens=i+1))
        yield dict(kind='eco_native_sse', request_id='r', at_s=11+i/10, payload=payload)
        yield dict(kind='eco_engine_output', request_id='r', token_index=i+1, token_ids=[i+100], at_s=12+i/10)
        yield dict(kind='eco_client_sse', request_id='r', at_s=13+i/10, payload=payload)
    yield dict(kind='eco_membership_commit', operation='remove', version=3, at_s=20,
               native_generation=5, drained=True, live_kv_blocks=0, energy_j=12.3456)


def test_roundtrip_tokens_timestamps_energy_and_actions_exact(tmp_path):
    source = list(rows(32))
    path = tmp_path/'events.jsonl.gz'
    with CompactJournal(path) as writer:
        for row in source:
            writer.write(row)
    assert list(iter_journal(path.with_suffix(''))) == source
    raw = list(iter_jsonl(path))
    assert sum('payload_data' in row for row in raw) == 32
    assert sum('token_ids' in row for row in raw) == 0
    assert all('text' not in row.get('payload_data', {}).get('body', {}) for row in raw)
    native = [r['payload'] for r in iter_journal(path) if r['kind']=='eco_native_sse']
    receipt = payload_receipt(native, journal_path=path.name, request_id='r')
    assert receipt['completion_tokens']==32 and receipt['terminal'] and 'events' not in receipt


def test_unicode_suffix_revisions_and_different_choice_text_are_lossless(tmp_path):
    source = [dict(kind='token', request_id='r', payload=dict(text=text, choices=[dict(text=choice)]))
              for text, choice in [('x�','a�'),('x你好','a你'),('x你好!','a你好')]]
    path=tmp_path/'events.jsonl.gz'
    with CompactJournal(path) as writer:
        for row in source:
            writer.write(row)
    assert list(iter_journal(path))==source


def test_payload_storage_is_linear_before_compression(tmp_path):
    sizes=[]
    for count in (128,512,2048):
        path=tmp_path/f'{count}.jsonl.gz'
        with CompactJournal(path) as writer:
            for row in rows(count):
                writer.write(row)
        with gzip.open(path,'rb') as handle:
            sizes.append(len(handle.read()))
    assert sizes[1]/sizes[0]<4.3 and sizes[2]/sizes[1]<4.3


def test_legacy_json_jsonl_and_truncated_gzip_fail_closed(tmp_path):
    source=list(rows(3))
    old=tmp_path/'old.jsonl'
    old.write_text(''.join(json.dumps(row)+'\n' for row in source))
    assert list(iter_journal(old))==source
    data=tmp_path/'data.json.gz'
    with gzip.open(data,'wt') as handle:json.dump(dict(ok=True),handle)
    assert read_json(data)==dict(ok=True)
    data.write_bytes(data.read_bytes()[:-5])
    with pytest.raises((EOFError, gzip.BadGzipFile)):
        read_json(data)
    path=tmp_path/'new.jsonl.gz'
    with CompactJournal(path) as writer:
        for row in source:writer.write(row)
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises((EOFError,gzip.BadGzipFile)):
        list(iter_journal(path))


def test_stage_checkpoint_is_readable_without_accepting_truncation(tmp_path):
    path=tmp_path/'events.jsonl.gz'
    source=list(rows(2))
    writer=CompactJournal(path)
    writer.write(source[0]);writer.checkpoint()
    assert list(iter_journal(path))==source[:1]
    for row in source[1:]:writer.write(row)
    writer.close()
    assert list(iter_journal(path))==source


def test_mixed_module_alias_pickle_and_monkeypatch_identity(monkeypatch):
    old=importlib.import_module('pdblend.bench.native_mixed')
    new=importlib.import_module('pdblend_baselines.mixed.run_native')
    policy=importlib.import_module('pdblend_baselines.mixed.policy')
    assert old is new
    assert importlib.import_module('pdblend_baselines.mixed_policy') is policy
    assert pickle.loads(b'cpdblend_baselines.mixed_policy\nMixedReplica\n.') is policy.MixedReplica
    replacement=object()
    monkeypatch.setattr(old,'generate',replacement)
    assert new.execute.__globals__['generate'] is replacement
