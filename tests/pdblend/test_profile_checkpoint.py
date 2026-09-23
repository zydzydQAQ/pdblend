import json
from types import SimpleNamespace

import pytest

from pdblend.profile.profiler import Profiler
from pdblend.profile.identity import sha256_value


def profiler(tmp_path):
    p = Profiler.__new__(Profiler)
    p.out_dir = tmp_path
    p.raw = {'schema': 2, 'prefill': [], 'decode': [], 'mixed': [], 'static': {}, 'transfer': []}
    return p


def test_interrupted_checkpoint_preserves_last_raw_and_bindings(tmp_path, monkeypatch):
    p = profiler(tmp_path)
    p._checkpoint()
    original = (tmp_path / 'raw.json').read_bytes()
    old = json.loads(original)
    p.raw['prefill'].append({'freq_mhz': 2100, 'input_tokens': 128, 'seconds': .1})

    def fail_replace(*args):
        raise OSError('interrupted publication')

    monkeypatch.setattr('pdblend.profile.profiler.os.replace', fail_replace)
    with pytest.raises(OSError, match='interrupted'):
        p._checkpoint()
    assert (tmp_path / 'raw.json').read_bytes() == original
    from pathlib import Path
    import hashlib
    for binding in old['evidence_bindings'].values():
        for name, checksum in binding['files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == checksum


def test_checkpoint_archive_is_not_claimed_as_independent_holdout(tmp_path):
    p = profiler(tmp_path)
    p._checkpoint()
    raw = json.loads((tmp_path / 'raw.json').read_text())
    assert raw['holdout_independent'] is False
    assert raw['identity_sha256'] == sha256_value({k: v for k, v in raw.items() if k != 'identity_sha256'})


def test_resume_rejects_tampered_raw_before_using_any_rows(tmp_path):
    p = profiler(tmp_path)
    p._checkpoint()
    raw = json.loads((tmp_path / 'raw.json').read_text())
    raw['decode'].append({'power_w': 999})
    (tmp_path / 'raw.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='invalid checkpoint digest'):
        p.resume()


def test_worker_heartbeat_does_not_invalidate_checkpoint_snapshot(tmp_path):
    import hashlib
    p = profiler(tmp_path)
    source = tmp_path / 'concurrency-environment.json'
    source.write_text(json.dumps({'peer_snapshots': [{'peers': []}]}))
    p._checkpoint()
    raw = json.loads((tmp_path / 'raw.json').read_text())
    bound = raw['concurrency_environment']
    source.write_text(json.dumps({'peer_snapshots': [{'peers': [{'job_id': 'next'}]}]}))
    assert hashlib.sha256((tmp_path / bound['samples_file']).read_bytes()).hexdigest() == bound['samples_sha256']
    p._checkpoint()
    assert p.raw['concurrency_environment']['samples_sha256'] != bound['samples_sha256']


def test_retry_mixed_point_preserves_failure_without_duplicate_grid_key(tmp_path):
    p = profiler(tmp_path)
    failed = dict(freq_mhz=2100, batch=8, chunk_tokens=512, valid=False)
    valid = dict(failed, valid=True)
    p._record_mixed(failed)
    p._record_mixed(valid)
    assert p.raw['mixed'] == [valid]
    assert p.raw['mixed_attempt_history'] == [failed]


def test_resume_rejects_changed_interference_receipt(tmp_path):
    import hashlib
    p = profiler(tmp_path)
    evidence = tmp_path / 'interference.json'
    evidence.write_text('{"passed":true}')
    p.raw['external_interference'] = {
        'samples_file': evidence.name,
        'samples_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    p._checkpoint()
    evidence.write_text('{"passed":false}')
    with pytest.raises(ValueError, match='invalid receipt: external_interference'):
        p.resume()


def test_completed_resume_does_not_reload_models(tmp_path, monkeypatch):
    p = profiler(tmp_path)
    p._finish_profile = lambda started: tmp_path / 'profile.json'

    def unexpected_fleet(*args, **kwargs):
        raise AssertionError('completed checkpoint must not reload models')

    monkeypatch.setattr('pdblend.profile.profiler.Fleet', unexpected_fleet)
    assert p.run(()) == tmp_path / 'profile.json'


def test_mixed_uses_readonly_inherited_references_without_relabelling(tmp_path):
    import asyncio
    import copy
    from contextlib import asynccontextmanager
    p = profiler(tmp_path)
    p.raw.update(system='pdblend', model_id='model', model_hash='m', tokenizer_hash='t', tp=4, pp=1)
    p.freqs = p.mixed_freqs = (1500,)
    p.parallel_layout = {}
    p._lock = lambda *args: None
    entered = []
    @asynccontextmanager
    async def background(client, batch, context, tag):
        entered.append((batch,context))
        raise RuntimeError('sampling-path-reached')
        yield
    p._background = background
    reference = copy.deepcopy(p.raw)
    reference['decode'] = [dict(freq_mhz=1500,batch=b,context_tokens=1024,step_seconds=.02,
        evidence_source='prior') for b in (8,32)]
    reference['prefill'] = [dict(freq_mhz=1500,input_tokens=n,seconds=.1,evidence_source='prior')
                            for n in (512,2048)]
    before = copy.deepcopy(reference)
    asyncio.run(p._mixed(None,[0,1,2,3],checkpoint=False,reference_raw=reference))
    assert len(entered)==4 and reference==before
    assert not p.raw['prefill'] and not p.raw['decode']
    assert all(row['invalid_reason']=='sampling-path-reached' and row['base_step_s']==.02
               and row['alone_prefill_s']==.1 and row['reference_binding']['decode_evidence_source']=='prior'
               for row in p.raw['mixed'])
    reference['model_id']='different-model'
    with pytest.raises(ValueError,match='identity differs'):
        asyncio.run(p._mixed(None,[0,1,2,3],checkpoint=False,reference_raw=reference))
