import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

PACKAGE=Path(__file__).resolve().parent.parent/'B32B-main-scale-fixed-window-v1'
sys.path.insert(0,str(PACKAGE))
s=importlib.util.spec_from_file_location('b_main_bound_child',PACKAGE/'child.py')
child=importlib.util.module_from_spec(s);s.loader.exec_module(child)
SPEC=json.loads((PACKAGE/'runspec.json').read_text())


@pytest.mark.parametrize('dataset',['alpaca','sharegpt','longbench'])
@pytest.mark.parametrize('scale',[.5,1.,2.])
def test_actual_frozen_controller_receives_dataset_scale(tmp_path,monkeypatch,dataset,scale):
    from ecopadg.serving import cell
    row=next(r for r in SPEC['cells'] if r['dataset']==dataset and r['slo_scale']==scale)
    args=child.cell_arguments(row);args.out=tmp_path/'out';seen=[]
    class BeforeNetwork(Exception):pass
    def constructor(cfg):seen.append(copy.deepcopy(cfg));raise BeforeNetwork()
    monkeypatch.setattr(cell,'Controller',constructor)
    with pytest.raises(BeforeNetwork):asyncio.run(cell.run_cell(args))
    cfg=seen[0]
    assert cfg['slo_scale']==scale
    assert (cfg['slo_ttft_s'],cfg['slo_tpot_s'])==(row['slo_ttft_s'],row['slo_tpot_s'])
    assert cfg['arrival_window_s']==300 and cfg['measurement_window_protocol']=='per-dataset-slo-fixed-window-v2'
    assert cfg['output_prior']==211 and cfg['allow_pd'] is False and cfg['dynamic_pools'] is False
    assert cfg['scheduler_budget_ablation']['max_num_batched_tokens']==8192
    assert args.timeout==120 and cfg['node_gpus']==list(range(8))


def test_exact_scale_work_reuse_and_two_seed_ten_rate_grid():
    main=[r for r in SPEC['cells'] if r['phase']=='main'];scale=[r for r in SPEC['cells'] if r['phase']=='scale']
    assert len(main)==60 and len(scale)==36 and len({r['reuse_main_cell_id'] for r in scale})==18
    idx={r['cell_id']:r for r in main}
    for r in scale:
        original=idx[r['reuse_main_cell_id']]
        assert all(r[k]==original[k] for k in ('trace','trace_sha256','n_requests','seed','dataset','rate_rps'))
    for dataset in ('alpaca','sharegpt','longbench'):
        group=[r for r in main if r['dataset']==dataset]
        assert len({r['rate_rps'] for r in group})==10
        assert {r['seed'] for r in group}=={701,1701}
