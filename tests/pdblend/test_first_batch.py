import hashlib
import json

import pytest

from pdblend.bench.first_batch import build, load_anchor, sha


def corpora(root):
    for size in ('7b','14b','32b'):
        path = root/f'2026-09-22-{size}-v1'; path.mkdir(parents=True)
        hashes = {}
        for dataset in ('alpaca','sharegpt','longbench'):
            p = path/f'{dataset}.json'
            p.write_text(json.dumps(dict(evaluation=[dict(prompt=[1,2,3], output_tokens=16)])))
            hashes[dataset] = sha(p)
        (path/'manifest.json').write_text(json.dumps(dict(complete=True,
            model_name=f'Qwen2.5-{size.upper()}-Instruct', tokenizer_sha256='tokenizer-'+size,
            dataset_sha256=hashes)))


def test_first_batch_is_45_distinct_decisions_without_topology_multiplication(tmp_path):
    corpora(tmp_path/'corpora')
    value = build(tmp_path/'out', tmp_path/'corpora')
    assert len(value['points']) == len({p['name'] for p in value['points']}) == 45
    assert {p['seed'] for p in value['points']} == {701}
    assert {p['duration_s'] for p in value['points']} == {300}
    assert all(p['trace'] is None and 'missing_rate_anchor' in p['blockers'] for p in value['points'])
    pd = [p for p in value['points'] if p['system']=='pdblend']
    assert len(pd) == 9 and all('tp' not in p['topology'] for p in pd)
    fixed32 = [p for p in value['points'] if '32B' in p['model_id'] and p['system'] in ('mixed','ecoserve')]
    assert all(p['topology']['tp'] == 2 for p in fixed32)


def test_shared_trace_frozen_once_and_anchor_tampering_is_rejected(tmp_path):
    corpus = tmp_path/'corpora'; corpora(corpus)
    path = tmp_path/'anchor'; path.mkdir()
    anchors = {}
    for dataset in ('alpaca','sharegpt','longbench'):
        confirmation = path/dataset/'completion.json'; confirmation.parent.mkdir()
        confirmation.write_text(json.dumps(dict(split='tuning', metrics=dict(passed=True))))
        anchors[dataset] = dict(base_rate_rps=2, scope='highest_tested_and_confirmed_passing_rate',
            corpus_sha256=sha(corpus/'2026-09-22-7b-v1'/f'{dataset}.json'),
            confirmation_path=f'/output/anchor/{dataset}/completion.json',
            confirmation_sha256=sha(confirmation))
    completion = path/'completion.json'
    completion.write_text(json.dumps(dict(status='passed', complete=True, model_id='Qwen2.5-7B-Instruct',
        evaluation_used_for_selection=False, selection_splits=['calibration','tuning'], anchors=anchors)))
    value = build(tmp_path/'out', corpus, {'7b':completion})
    rows = [p for p in value['points'] if p['model_id']=='Qwen2.5-7B-Instruct' and p['dataset']=='alpaca']
    assert len({p['trace']['sha256'] for p in rows}) == 1
    assert all(p['rate_rps'] == 1 and not p['formal_eligible'] for p in rows)
    (path/'alpaca/completion.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_anchor(completion,'Qwen2.5-7B-Instruct','alpaca',anchors['alpaca']['corpus_sha256'])
