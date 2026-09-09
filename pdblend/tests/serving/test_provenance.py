from copy import deepcopy
from pathlib import Path
import json

import pytest
from ecopadg.serving import provenance
from ecopadg.serving.evidence import freeze_files


def test_live_process_cannot_reuse_stale_or_partial_source_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(provenance,'MODEL_ROOT',tmp_path)
    for name in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors'):
        (tmp_path/name).write_text('fixture')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model.safetensors'}}))
    files=freeze_files(Path(provenance.__file__).parent.glob('*.py'))
    freeze=dict(files=files,groups=dict(source=list(files),model=[str(p) for p in tmp_path.iterdir()]))
    config=dict(tp=2,gpus=[0,1])
    raw=dict(instance_id='i1',tp=2,engine_version='0.9.2',dtype='bfloat16',
             max_model_len=8192,cuda_visible_devices='0,1',source_files_at_import=files,
             model='/models/Qwen2.5-14B-Instruct')
    provenance.verify_engine_source(raw,'i1',config,freeze)
    stale=deepcopy(raw)
    stale['source_files_at_import'][next(iter(files))]='old process hash'
    with pytest.raises(ValueError,match='running engine differs'):
        provenance.verify_engine_source(stale,'i1',config,freeze)
    partial=deepcopy(freeze)
    partial['groups']['source'].pop()
    with pytest.raises(ValueError,match='every serving source'):
        provenance.verify_engine_source(raw,'i1',config,partial)
    with pytest.raises(ValueError,match='running engine differs'):
        provenance.verify_engine_source(raw,'i1',dict(tp=2,gpus=[2,3]),freeze)
    with pytest.raises(ValueError,match='fixed 14B model'):
        provenance.verify_engine_source(dict(raw,model='/models/other'),'i1',config,freeze)
    incomplete=deepcopy(freeze);incomplete['groups']['model'].remove(str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError,match='weights and tokenizer'):
        provenance.verify_engine_source(raw,'i1',config,incomplete)
