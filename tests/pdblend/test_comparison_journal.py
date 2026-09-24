import gzip
import json

import pytest

from pdblend.bench.comparison_journal import JournalReadStats, iter_comparison_journal
from pdblend.bench.comparison_metrics import canonical_outcomes
from pdblend.results.journal import SCHEMA, CompactJournal, iter_journal


def write_compact(path, rows):
    with CompactJournal(path) as journal:
        for row in rows:
            journal.write(row)


def sample_rows(count=8):
    history = []
    for i in range(count):
        for rid in ('r0', 'r1', 'r2'):
            text = '你🙂' * i + ('�' if i % 2 == 0 else '好')
            payload = dict(text=text, choices=[dict(text=text), dict(text='选项'+text)],
                           token_ids=[100+i], token_index=i+1, received_s=101+i,
                           finished=i==count-1)
            history.append(dict(kind='mixed_native_sse', request_id=rid, payload=payload))
            yield history[-1]
            yield dict(kind='mixed_engine_output', request_id=rid, token_index=i+1, token_ids=[100+i])
            yield dict(kind='mixed_client_sse', request_id=rid, payload=payload)
    # Older refs force replay after evictions; native/client share definitions.
    yield from history[::7]
    yield dict(kind='action', generation=3, at_s=300, energy_j=123.456)


@pytest.mark.parametrize('entries,characters', [(0,0), (1,1), (2,30), (16,100_000)])
def test_equivalent_unicode_shared_refs_eviction_and_budget(tmp_path, entries, characters):
    source = list(sample_rows())
    path = tmp_path/'events.jsonl.gz'
    write_compact(path, source)
    stats = JournalReadStats()
    decoded = list(iter_comparison_journal(path, max_cache_entries=entries,
                    max_cache_characters=characters, stats=stats))
    assert decoded == list(iter_journal(path)) == source
    assert stats.rows == len(source)
    assert stats.cache_peak_entries <= entries
    assert stats.cache_peak_characters <= characters


def test_cached_linear_chain_and_canonical_adapter_are_equivalent(tmp_path):
    source = list(sample_rows(48))
    # Exclude deliberately repeated old client-independent observations.
    path = tmp_path/'events.jsonl.gz'
    write_compact(path, source)
    stats = JournalReadStats()
    decoded = list(iter_comparison_journal(path, stats=stats))
    assert decoded == list(iter_journal(path))
    assert stats.patch_definitions_visited == 3*48
    assert stats.cache_hits >= 3*48
    trace = [dict(idx=i, arrival_s=0, output_tokens=48) for i in range(3)]
    outcomes = [dict(request_id='r'+str(i), completion_tokens=48, finished_s=149) for i in range(3)]
    assert canonical_outcomes('mixed', trace, outcomes, service_started_s=100,
                             journal=iter_comparison_journal(path)) == canonical_outcomes(
        'mixed', trace, outcomes, service_started_s=100, journal=iter_journal(path))


def definition(ref, previous=None, patch=None):
    return dict(_journal_schema=SCHEMA, payload_ref=ref,
                payload_data=dict(previous=previous, text_patches=patch or {},
                                  body=dict(token_ids=[7])))


@pytest.mark.parametrize('rows,error', [
    ([dict(_journal_schema=SCHEMA, payload_ref='absent')], 'unresolved compact payload'),
    ([dict(_journal_schema=SCHEMA, token_payload_ref='absent')], 'unresolved compact token'),
    ([definition('a', 'a')], 'invalid compact payload text chain'),
    ([definition('a', 'missing')], 'invalid compact payload text chain'),
    ([definition('a'), definition('b', 'c'), definition('c', 'b')], 'invalid compact payload text chain'),
    ([definition('a'), definition('a')], 'duplicate payload definition'),
    ([definition('a', patch={'text': [1, 'x']})], 'invalid compact text prefix'),
    ([definition('a', patch={'text': [0, 'x']}),
      definition('b', 'a', {'text': [True, 'y']})], 'invalid compact text prefix'),
    ([dict(_journal_schema='unknown')], 'unsupported compact journal schema'),
])
def test_invalid_evidence_rejected_even_with_cached_prefix(tmp_path, rows, error):
    path = tmp_path/'bad.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for reader in (iter_journal, iter_comparison_journal):
        with pytest.raises(ValueError, match=error):
            list(reader(path))


def test_legacy_concatenated_gzip_and_truncation(tmp_path):
    legacy = tmp_path/'legacy.jsonl'
    legacy.write_text('\n'+json.dumps(dict(payload={'text':'你好'}, token_ids=[12]))+'\n')
    assert list(iter_comparison_journal(legacy)) == list(iter_journal(legacy))
    path = tmp_path/'events.jsonl.gz'
    source = list(sample_rows(2))
    with CompactJournal(path) as writer:
        for row in source[:5]:writer.write(row)
        writer.checkpoint()
        for row in source[5:]:writer.write(row)
    assert list(iter_comparison_journal(path.with_suffix(''))) == source
    path.write_bytes(path.read_bytes()[:-5])
    for reader in (iter_journal, iter_comparison_journal):
        with pytest.raises((EOFError, gzip.BadGzipFile)):
            list(reader(path))
