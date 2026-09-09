"""Exercise actual queue selection without touching hardware or rewriting rows."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('actual_group_runner', ROOT/'run.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


@pytest.mark.parametrize('datasets,expected',[(None,30),(['alpaca','sharegpt'],20),(['longbench'],10)])
def test_real_sweep_selects_original_main_rows(monkeypatch,tmp_path,datasets,expected):
    import ecopadg.measure.backends as backends
    source=ROOT.parent/'five-system-fixed-window-v1/sources/A14B/manifest.json'
    original=runner.read(source);seen=[]
    configs={d:'not-used-in-fixture' for d in ('alpaca','sharegpt','longbench')}
    binding=dict(model='14b',system='distserve',configs=configs,output=str(tmp_path/'results'),
        files={str(source.resolve()):runner.sha(source)})
    monkeypatch.setattr(runner,'validate_binding',lambda binding:None)
    monkeypatch.setattr(backends,'PynvmlBackend',lambda **kw:object())
    async def to_thread(fn,*args,**kwargs):return fn(*args,**kwargs)
    monkeypatch.setattr(runner.asyncio,'to_thread',to_thread)
    async def cell(session,b,row,output,hardware):
        seen.append(row)
        path=output/'operations'/row['cell_id']/'receipt.json'
        receipt=dict(measurement_valid=True,summary=dict(work_complete=False))
        runner.write(path,receipt)
        return receipt
    monkeypatch.setattr(runner,'run_one',cell)
    bpath=tmp_path/'binding.json';runner.write(bpath,binding)
    args=SimpleNamespace(manifest=source,binding=bpath,system='distserve',phase='main',dataset=datasets,max_cells=30)
    asyncio.run(runner.sweep(args,binding))
    selected=set(datasets or configs)
    assert len(seen)==expected
    assert seen==[r for r in original['cells'] if r['system']=='distserve' and r['phase']=='main' and r['dataset'] in selected]
    assert runner.sha(source)==binding['files'][str(source.resolve())]


def test_group_cannot_request_unbound_dataset(monkeypatch,tmp_path):
    source=ROOT.parent/'five-system-fixed-window-v1/sources/A14B/manifest.json'
    binding=dict(model='14b',system='distserve',configs={'longbench':'x'},
        files={str(source.resolve()):runner.sha(source)})
    monkeypatch.setattr(runner,'validate_binding',lambda binding:None)
    args=SimpleNamespace(manifest=source,system='distserve',phase='main',dataset=['alpaca'])
    with pytest.raises(RuntimeError,match='does not bind'):
        asyncio.run(runner.sweep(args,binding))
