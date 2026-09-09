"""CPU only: actual new scope, original loader/binding code and prior-attempt gates."""
import copy,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace as N
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import technical as t,common as c,run as r
from test_child import child,job


def test_scope_only_changes_two_observed_UUIDs():
    new=c.observation_declaration();old=c.read(c.CANDIDATE/'specs/original-vs-continuous.json')
    assert new['requests']==old['requests'] and {k for k in old if old[k]!=new[k]}=={'request_ids'}
    assert new['request_ids']==[new['requests'][1]['request_uuid'],new['requests'][3]['request_uuid']]


def test_actual_compact_engine_loader_and_original_six_request_child_contract(tmp_path):
    _,j,s=job(tmp_path);p=Path(j['observation_spec']);assert len(p.read_bytes())>16384
    with pytest.raises(RuntimeError,match='loader rejected'):t.spec_preflight(p)
    t.compact_spec(p,s);proof=t.spec_preflight(p);assert proof['passed'] and proof['bytes']<16384 and not proof['writer_created'] and c.read(p)==s
    j['observation_spec_sha256']=c.sha(p);assert child.validate_job(j)==s


def test_new_scope_reaches_actual_hook_UUID_selection(monkeypatch):
    sys.path.insert(0,str(c.CANDIDATE))
    sp=importlib.util.spec_from_file_location('original_hook_cpu_fixtures',c.CANDIDATE/'test_observation.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
    state=f.d.State(c.observation_declaration(),'f'*64);state.writer=f.Sink();monkeypatch.setattr(f.d,'_STATE',state);monkeypatch.setattr(f.d,'_LOADED',True)
    golden,temporal=c.observation_declaration()['request_ids'];continuous=c.observation_declaration()['requests'][5]['request_uuid']
    g,mi,sm=f.fixture_input(golden,other_first=False)
    # Actual solo decode shape is one row, still sampled through the original binding code.
    mi.request_ids_to_seq_ids={golden:[17]};mi.query_lens=[1];sm.seq_groups=sm.seq_groups[:1]
    mi.input_tokens=mi.input_tokens[:1];mi.input_positions=mi.input_positions[:1];a=mi.attn_metadata
    for field in ('block_tables','slot_mapping','seq_lens_tensor','context_lens_tensor'):setattr(a,field,getattr(a,field)[:1])
    a.query_start_loc=f.Tensor([0,1]);a.num_decode_tokens=1
    bound=f.d.bind([g],mi,sm,16);assert bound['records'][0]['request_id']==golden and bound['records'][0]['input_sequence_row']==0 and bound['records'][0]['logits_row']==0
    mi.diagnostic_bindings=bound;monkeypatch.setitem(sys.modules,'torch',f.torch_fake());assert f.d.begin(mi,0,2,1,True)
    g,mi,sm=f.fixture_input(temporal);assert f.d.bind([g],mi,sm,16)['records'][0]['request_id']==temporal
    g,mi,sm=f.fixture_input(continuous);assert f.d.bind([g],mi,sm,16) is None
    assert not state.error


def test_only_exact_two_retained_claims_and_valid_restoration():
    a=c.read(t.AUTH);assert t.verify_evidence(a,a['prior_claims'],lambda _:False)==a

@pytest.mark.parametrize('mutation',['extra_claim','missing_claim','prior_live','wrong_scope','third_claim','wrong_binding_sha'])
def test_unreviewed_or_live_prior_cannot_pass(mutation):
    a=copy.deepcopy(c.read(t.AUTH));claims=a['prior_claims'].copy();live=lambda _:False
    if mutation=='extra_claim':claims.append('/foreign/claim.json')
    if mutation=='missing_claim':claims.pop()
    if mutation=='prior_live':live=lambda _:True
    if mutation=='wrong_scope':a['attempt_name']='B32B-temporal-observation-attempt-004'
    if mutation=='third_claim':claims.append(str(c.ROOT/'execution-once.json'))
    if mutation=='wrong_binding_sha':a['files'][a['restored_binding']]='0'*64
    with pytest.raises(RuntimeError):t.verify_evidence(a,claims,live)


def test_old_scope_capture_cannot_be_reused_for_new_solo_observation(tmp_path):
    sys.path.insert(0,str(c.CANDIDATE))
    sp=importlib.util.spec_from_file_location('unchanged_offline_verifier_scope_test',c.CANDIDATE/'verify_capture.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
    path=tmp_path/'new-spec.json';t.compact_spec(path,c.observation_declaration());old=c.ROOT.parent/'B32B-temporal-observation-attempt-002/results'
    with pytest.raises(ValueError):v.verify(path,old/'capture-live',old/'child/full-outputs.json')


def test_actual_frozen_capture_import_closes_without_candidate_search_path(tmp_path,monkeypatch):
    old=c.ROOT.parent/'B32B-temporal-observation-attempt-002/results';state=c.read(old/'status.json')
    monkeypatch.setattr(sys,'path',[x for x in sys.path if x!=str(c.CANDIDATE)])
    monkeypatch.delitem(sys.modules,'pdblend_diagnostics',raising=False)
    result=c.frozen_capture(old.parent/'observation-spec.json',old/'capture-live',tmp_path/'frozen',old/'child/full-outputs.json',state['child_status'],state['process_terminal'])
    assert result['capture_complete'] and len(result['records'])==14
    assert 'pdblend_diagnostics' not in sys.modules
