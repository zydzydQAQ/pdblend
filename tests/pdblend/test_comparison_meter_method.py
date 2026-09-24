"""The reusable meter gate preserves the frozen raw-method checks."""
import ast
from pathlib import Path

import pytest

from pdblend.bench.comparison_meter_method import audit_isolated_meter_method
from test_comparison_ecoserve import isolated_method_fixture, put


def test_common_gate_preserves_frozen_ecoserve_method_semantics():
    root=Path(__file__).parents[2]/'src/pdblend/bench'
    def normalized(name, function):
        tree=ast.parse((root/name).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        node.name='method'
        return ast.dump(node,include_attributes=False)
    assert normalized('comparison_meter_method.py','audit_isolated_meter_method') == normalized(
        'comparison_ecoserve_acceptance.py','_isolated_meter_method')


@pytest.mark.parametrize('stopped',[False,True])
def test_shared_gate_replays_actual_startup_and_full_service_tail(tmp_path,stopped):
    args,method=isolated_method_fixture(tmp_path,stopped=stopped)
    args['point']['system']='distserve'
    assert audit_isolated_meter_method(**args)==method


@pytest.mark.parametrize('fault',['same_process','rpc_in_tail','source_changed','missing_guard','mode'])
def test_shared_gate_rejects_invalid_method_even_when_power_is_complete(tmp_path,fault):
    args,method=isolated_method_fixture(tmp_path)
    if fault=='same_process':method['child_pid']=method['parent_pid']
    elif fault=='rpc_in_tail':method['commands'][-1]['requested_s']=249.
    elif fault=='source_changed':method['public_sampler']['sha256']='bad'
    elif fault=='missing_guard':method['local_window_guards']=[]
    elif fault=='mode':args['identity']['metering_execution']='in_process'
    args['raw_refs']['metering_method']=put(tmp_path,'metering-method.json',method)
    with pytest.raises(ValueError):audit_isolated_meter_method(**args)
