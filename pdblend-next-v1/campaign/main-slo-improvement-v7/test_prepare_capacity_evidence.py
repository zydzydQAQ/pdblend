import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import prepare_capacity_evidence as e


def fixture(tmp_path):
    host=tmp_path/'host';host.mkdir();(host/'manifest.json').write_text('{}')
    trace={'path':'/trace1','sha256':'1'*64}
    spec=dict(original_binding={'path':'/original','sha256':'o'},
        capacity_binding={'path':'/capacity','sha256':'c'},config={'path':'/config','sha256':'a'},
        host_release=str(host),demand_domain_sha256='d',cycles=[{'high2':trace}],matched_idle_duration_s=60.)
    directory=tmp_path/'measured/cycle-1-high2-layout2';directory.mkdir(parents=True)
    result=dict(phase=directory.name,source=dict(original_binding=spec['original_binding'],
        capacity_binding=spec['capacity_binding'],config=spec['config'],
        host_manifest=e.p.ref(host/'manifest.json')),trace=trace,demand_domain_sha256='d')
    path=directory/'result.json';path.write_text(json.dumps(result))
    return spec,path,result


def test_exact_phase_is_retained(tmp_path):
    spec,path,result=fixture(tmp_path)
    assert e.measured_result(spec,path.parents[1],1,'high2-layout2')==e.p.ref(path)


@pytest.mark.parametrize('field,value',[('config',{'path':'/other','sha256':'b'}),
                                      ('capacity_binding',{'path':'/other','sha256':'c'})])
def test_same_domain_cannot_mix_sources(tmp_path,field,value):
    spec,path,result=fixture(tmp_path);result['source'][field]=value
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='source/config'):
        e.measured_result(spec,path.parents[1],1,'high2-layout2')


def test_cannot_replace_declared_repeat_with_another_trace(tmp_path):
    spec,path,result=fixture(tmp_path);result['trace']={'path':'/trace2','sha256':'2'*64}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='trace/domain'):
        e.measured_result(spec,path.parents[1],1,'high2-layout2')


def test_import_cache_cannot_silently_use_other_version(tmp_path,monkeypatch):
    for name in ('capacity_executor','capacity_backend','capacity_certificate','capacity_runtime','capacity_planner'):
        (tmp_path/(name+'.py')).write_text('# qualified implementation\n')
    old=tmp_path/'other.py';old.write_text('# obsolete\n')
    monkeypatch.setitem(sys.modules,'capacity_executor',SimpleNamespace(__file__=str(old)))
    with pytest.raises(ValueError,match='conflicting imported'):
        e.import_code(tmp_path,'capacity_certificate')
