import json

import pytest

from ecopadg.scalability.native_logs import begin_capture, finish_capture, summarize_kv


def test_capture_keeps_only_append_and_detects_truncation(tmp_path):
    path=tmp_path/'engine.jsonl'
    path.write_text('old-line\n')
    config=dict(instances=[dict(id='p')],native_kv_logs={'p':str(path)})
    cursors=begin_capture(config)
    with path.open('a') as f:
        f.write(json.dumps(dict(engine_id='p',rank=0,direction='send'))+'\n')
    out=tmp_path/'out';out.mkdir()
    rows=finish_capture(cursors,out)
    assert len(rows)==1 and rows[0]['engine_id']=='p'
    path.write_text('')
    with pytest.raises(ValueError,match='truncated'):
        finish_capture(cursors,tmp_path)


def test_kv_pairs_real_nonces_excludes_warmup_and_counts_batch_once():
    controls=[dict(kind='admission',request_id=rid,at_s=t,
        plan=dict(routes=[dict(prefill_id='p',decode_id='d')])) for rid,t in [('warm',1),('a',10),('b',10)]]
    rows=[dict(direction='send',request_ids=['pdb:a:p:p:d','pdb:b:p:p:d'],started_s=11,finished_s=12),
          dict(direction='receive',request_ids=['pdb:a:d:p:d','pdb:b:d:p:d'],started_s=13,finished_s=14),
          dict(direction='send',request_ids=['pdb:warm:p:p:d'],started_s=11,finished_s=19)]
    result=summarize_kv(rows,controls,10,20)
    assert result['kv_transfer_count']==2
    assert result['kv_transfer_s']['p99']==3
    assert result['kv_send_s']['p99']==1
    assert result['kv_receive_s']['p99']==1
    assert result['kv_unpaired_pd_requests']==0


def test_missing_log_configuration_rejected(tmp_path):
    with pytest.raises(ValueError,match='path missing'):
        begin_capture(dict(instances=[dict(id='p')]))
