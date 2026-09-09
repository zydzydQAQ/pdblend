import json
import time
from types import SimpleNamespace

from ecopadg.measure import backends
from ecopadg.serving import campaign
import pytest


def test_deadline_cleanup_is_scoped_and_restores_all_authorized_clocks(tmp_path,monkeypatch):
    calls=[];restored=[]
    def command(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='pdb-v2-i30\nunrelated-model\npdb-v2-bad;name\n')
    class Hardware:
        def reset_clock(self,gpu): restored.append(gpu)
    monkeypatch.setattr(campaign.subprocess,'run',command)
    monkeypatch.setattr(backends,'PynvmlBackend',Hardware)
    c=campaign.Campaign(tmp_path/'campaign')
    c.state['started_s']=time.time()-86400+40
    c.close()
    assert calls[-1]==['docker','rm','-f','pdb-v2-i30']
    assert restored==list(range(8))
    proof=json.loads((tmp_path/'campaign'/'deadline_cleanup.json').read_text())
    assert proof['stopped']==['pdb-v2-i30'] and not proof['errors']


def test_cpu_stage_cannot_extend_an_active_node_lease_past_deadline(tmp_path,monkeypatch):
    timeouts=[]
    monkeypatch.setattr(campaign.subprocess,'Popen',lambda *a,**kw:
        SimpleNamespace(wait=lambda timeout:timeouts.append(timeout) or 0))
    c=campaign.Campaign(tmp_path/'campaign')
    try:
        c.state['started_s']=time.time()-86400+80
        c.run('build',['python3','build.py'],120,gpu=False)
        assert 0<timeouts[0]<=20
        c.state['started_s']=time.time()-86400+59
        with pytest.raises(TimeoutError): c.run('late-build',['python3','build.py'],120,gpu=False)
    finally:
        c.state['started_s']=None
        c.close()
