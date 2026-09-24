#!/usr/bin/env python3
"""Small synthetic CPU-only equivalence benchmark; never reads GPU evidence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time

from pdblend.bench.comparison_journal import JournalReadStats, iter_comparison_journal
from pdblend.bench.resident_session import write_new
from pdblend.results.journal import CompactJournal, iter_journal


ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def consume(reader):
    digest, rows = hashlib.sha256(), 0
    started = time.perf_counter()
    for row in reader:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
        rows += 1
    return dict(elapsed_s=time.perf_counter()-started, rows=rows, decoded_sha256=digest.hexdigest())


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix='pdblend-journal-synthetic-') as temporary:
        for count in (128, 512, 1024):
            path = Path(temporary)/f'{count}.jsonl.gz'
            with CompactJournal(path) as writer:
                for i in range(count):
                    for rid in ('r0', 'r1'):
                        text = '你🙂'*i+('�' if i%2 == 0 else '好')
                        payload = dict(text=text, choices=[dict(text=text)], token_ids=[i],
                                       token_index=i+1, received_s=100+i/1000, finished=i==count-1)
                        writer.write(dict(kind='mixed_native_sse', request_id=rid, payload=payload))
                        writer.write(dict(kind='mixed_engine_output', request_id=rid,
                                          token_index=i+1, token_ids=[i]))
                        writer.write(dict(kind='mixed_client_sse', request_id=rid, payload=payload))
            reference = consume(iter_journal(path))
            stats = JournalReadStats()
            candidate = consume(iter_comparison_journal(path, stats=stats))
            if (reference['rows'], reference['decoded_sha256']) != (candidate['rows'], candidate['decoded_sha256']):
                raise ValueError('decoded output differs from frozen reference reader')
            results.append(dict(tokens_per_request=count, request_count=2, artifact_bytes=path.stat().st_size,
                reference=reference, candidate=candidate, cache_stats=asdict(stats),
                measured_speedup=reference['elapsed_s']/candidate['elapsed_s']))
    output = ROOT/'results/2026-09-24/comparison-journal-preflight/benchmark.json'
    write_new(output, dict(schema='comparison-journal-synthetic-benchmark-v1', hardware_executed=False,
        real_journals_read=False, frozen_execution_changed=False, formal_eligible=False,
        measured_at_s=time.time(), measurements=results, sources={str(p.relative_to(ROOT)):fingerprint(p)
            for p in (Path(__file__).resolve(), ROOT/'src/pdblend/bench/comparison_journal.py',
                      ROOT/'src/pdblend/results/journal.py',ROOT/'tests/pdblend/test_comparison_journal.py')},
        limits=['Synthetic CPU timing includes gzip/JSON/digest; not a formal-window speed claim.',
                'Definition storage is unchanged; the additional text-state cache is bounded.',
                'No runtime or frozen snapshot has been wired to this optional reader.']))
    print(json.dumps(dict(output=str(output), results=[dict(tokens=x['tokens_per_request'],
        speedup=x['measured_speedup'], reference_s=x['reference']['elapsed_s'],
        candidate_s=x['candidate']['elapsed_s']) for x in results])))


if __name__ == '__main__':
    main()
