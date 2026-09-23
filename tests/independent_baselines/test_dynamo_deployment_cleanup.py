import pytest
from types import SimpleNamespace

from pdblend_baselines.dynamollm import deployment


def _lifecycle(tmp_path):
    life = deployment.SubprocessLifecycle(
        dict(base_port=19000, model_id='Qwen2.5-7B-Instruct', legal_tp=[1, 2, 4],
             node_gpus=[0, 1, 2]), SimpleNamespace(instances={}), lambda *a, **k: None,
        tmp_path)
    process = SimpleNamespace(pid=77, returncode=0)
    life.processes['x'] = process
    life.instances['x'] = {'id': 'x', 'gpus': [0], 'port': 19000}
    life.transport.instances['x'] = life.instances['x']
    life.logs['x'] = SimpleNamespace(close=lambda: None)
    return life


def test_proc_group_state_distinguishes_zombies(monkeypatch):
    monkeypatch.setattr(deployment, '_proc_stat_entries', lambda: [
        (101, 'Z', 77), (102, 'S', 77), (103, 'S', 88)])
    state = deployment.proc_group_state(77)
    assert [row['pid'] for row in state['zombie_members']] == [101]
    assert [row['pid'] for row in state['live_members']] == [102]
    assert not state['no_live_members'] and not state['all_pid_gone']

    monkeypatch.setattr(deployment, '_proc_stat_entries', lambda: [(101, 'Z', 77)])
    state = deployment.proc_group_state(77)
    assert state['no_live_members'] and not state['all_pid_gone']

    monkeypatch.setattr(deployment, '_proc_stat_entries', lambda: [])
    state = deployment.proc_group_state(77)
    assert state['no_live_members'] and state['all_pid_gone']


def test_proc_group_state_fails_closed_on_proc_read_error(monkeypatch):
    def broken():
        raise RuntimeError('cannot read process stat')
    monkeypatch.setattr(deployment, '_proc_stat_entries', broken)
    with pytest.raises(RuntimeError, match='cannot read'):
        deployment.proc_group_state(77)


@pytest.mark.asyncio
async def test_stop_only_zombie_succeeds_with_no_live_receipt(tmp_path, monkeypatch):
    life = _lifecycle(tmp_path)
    monkeypatch.setattr(deployment, 'proc_group_state', lambda pgid: {
        'pgid': pgid, 'members': [{'pid': 88, 'state': 'Z', 'pgid': pgid}],
        'live_members': [], 'zombie_members': [{'pid': 88, 'state': 'Z', 'pgid': pgid}],
        'no_live_members': True, 'all_pid_gone': False})
    receipt = await life.stop('x')
    assert receipt['absence_scope'] == 'no_live_execution'
    assert receipt['no_live_members'] and not receipt['all_pid_gone']
    assert 'x' not in life.processes


@pytest.mark.asyncio
async def test_stop_live_then_exit_kills_and_records_final_state(tmp_path, monkeypatch):
    life = _lifecycle(tmp_path)
    states = iter([
        {'pgid': 77, 'members': [{'pid': 77, 'state': 'S', 'pgid': 77}],
         'live_members': [{'pid': 77, 'state': 'S', 'pgid': 77}], 'zombie_members': [],
         'no_live_members': False, 'all_pid_gone': False},
        {'pgid': 77, 'members': [], 'live_members': [], 'zombie_members': [],
         'no_live_members': True, 'all_pid_gone': True},
    ])
    monkeypatch.setattr(deployment, 'proc_group_state', lambda pgid: next(states))
    killed = []
    monkeypatch.setattr(deployment.os, 'killpg', lambda pgid, sig: killed.append((pgid, sig)))
    receipt = await life.stop('x')
    assert killed and receipt['all_pid_gone']


@pytest.mark.asyncio
async def test_stop_proc_error_keeps_owned_process(tmp_path, monkeypatch):
    life = _lifecycle(tmp_path)
    monkeypatch.setattr(deployment, 'proc_group_state', lambda pgid: (_ for _ in ()).throw(
        RuntimeError('stat permission denied')))
    with pytest.raises(RuntimeError, match='permission'):
        await life.stop('x')
    assert 'x' in life.processes and 'x' in life.instances


@pytest.mark.asyncio
async def test_stop_persistent_live_group_keeps_ownership(tmp_path, monkeypatch):
    life = _lifecycle(tmp_path)
    live = {'pgid': 77, 'members': [{'pid': 77, 'state': 'S', 'pgid': 77}],
            'live_members': [{'pid': 77, 'state': 'S', 'pgid': 77}], 'zombie_members': [],
            'no_live_members': False, 'all_pid_gone': False}
    monkeypatch.setattr(deployment, 'proc_group_state', lambda pgid: live)
    monkeypatch.setattr(deployment.os, 'killpg', lambda *a: None)
    monkeypatch.setattr(deployment, 'PROCESS_GROUP_GRACE_S', 0)
    with pytest.raises(RuntimeError, match='absence unconfirmed'):
        await life.stop('x')
    assert 'x' in life.processes and 'x' in life.instances
