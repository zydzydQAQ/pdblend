"""CPU-only exact B runtime/queue counterexamples; no hardware or policy change."""
import asyncio,importlib.util,json,sys,time,hashlib,ast,inspect
from pathlib import Path
import pytest
REPO=Path(__file__).resolve().parents[3]
PACKAGE=REPO/'campaign/cooperative-admission-yield-v1'
sys.path.insert(0,str(PACKAGE))
import cooperative_b_reproduce as proof
proof.SOURCE=REPO/'releases/five-system100-B32B-v1-runtime/src/ecopadg/serving'
import test_cooperative as shared
shared.proof=proof
CANDIDATE=REPO/'releases/five-system100-B32B-baseline-cooperative-v1-runtime'

def test_exact_b_ready_dispatch_fairness():shared.test_actual_dispatch_remains_cooperative_under_infeasible_ready_pressure()
def test_exact_b_fifo_capacity():shared.test_fifo_and_full_capacity_unchanged()
def test_exact_b_defer_deadline_sequence():shared.test_retry_deadline_and_original_sequence_preserved()
def test_exact_b_cancel_before_reserve():shared.test_cancel_at_new_yield_does_not_pop_or_reserve_request()
def test_exact_b_empty_arrival_removed():shared.test_wait_empty_then_arrival_and_remove_before_selection()
def test_exact_b_all_dispatch_paths_and_ast():shared.test_only_one_default_queue_yield_added_and_all_outer_branches_reenter_get()
def test_timeout_waiting_queue_leaves_no_inflight():
 async def run():
  q=shared.queue()(2)
  with pytest.raises(asyncio.TimeoutError):await asyncio.wait_for(q.get(),.001)
  assert not q.inflight and q.qsize()==0
  q.put_nowait('kept');assert await q.get()=='kept';q.defer('kept',.1)
  with pytest.raises(asyncio.TimeoutError):await asyncio.wait_for(q.get(),.001)
  assert not q.inflight and q.qsize()==1
  q.done('kept');assert q.qsize()==0 and not q.entries
 asyncio.run(run())
def test_actual_b_release_all_frozen_files_exact():
 parent=proof.SOURCE.parents[2];old=json.loads((parent/'manifest.json').read_text());new=json.loads((CANDIDATE/'manifest.json').read_text())
 assert set(old['files'])==set(new['files']) and len(old['files'])==133
 assert [p for p,h in old['files'].items() if new['files'][p]!=h]==['src/ecopadg/serving/admission.py']
 assert new['parent_manifest_sha256']==proof.sha(parent/'manifest.json')
 assert all(proof.sha(CANDIDATE/p)==h for p,h in new['files'].items())
 assert proof.sha(CANDIDATE/'src/ecopadg/serving/admission.py')==proof.sha(PACKAGE/'admission.py')
 assert len([p for p in old['files'] if p.endswith('.py')])==132
