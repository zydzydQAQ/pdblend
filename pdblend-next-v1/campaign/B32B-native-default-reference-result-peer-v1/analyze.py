"""Independent actual native/003 comparison; no hardware or source changes."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/root/workspace/pdblend-next-v1')
OUT = Path(__file__).resolve().parent
NATIVE = ROOT / 'campaign/B32B-native-default-reference-attempt-001'
OLD = ROOT / 'campaign/B32B-temporal-observation-attempt-003'
FILES = {}


def raw(path):
    path = Path(path).resolve(); data = path.read_bytes()
    FILES[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def read(path): return json.loads(raw(path))
def lines(path): return [json.loads(x) for x in raw(path).splitlines()]


def diff(a, b):
    assert len(a) == len(b)
    return next((dict(position_one_based=k, reference=x, observed=y)
        for k, (x, y) in enumerate(zip(a, b), 1) if x != y), None)


def capture(base, tokens, ids, owner):
    spec = read(base / 'observation-spec.json'); expected = {(rid, k) for rid in spec['request_ids'] for k in range(28, 35)}
    steps = {rid: [e for e in owner if rid in e['request_ids']] for rid in ids.values()}
    assert all(len(v) == 64 for v in steps.values())
    ranks = {}
    for rank in (0, 1):
        paths = list((base / 'results/capture-frozen').glob(f'rank{rank}-pid*.jsonl')); assert len(paths) == 1
        path = paths[0]; rows = lines(path); status = read(path.with_suffix('.status.json'))
        assert len(rows) == status['written'] == status['expected_records'] == 14
        assert status['complete'] and not status['failed'] and status['error'] is None
        for p in (path, path.with_suffix('.status.json')):
            assert raw(base / 'results/capture-live' / p.name) == p.read_bytes()
        indexed = {}
        for line, r in enumerate(rows, 1):
            rid, k = r['request_id'], r['output_index']; key = rid, k
            assert key not in indexed and key in expected
            assert r['rank'] == rank == status['rank'] and r['pid'] == status['pid']
            assert r['spec_sha256'] == FILES[str(base / 'observation-spec.json')] and r['observer_error'] is None
            assert r['tp'] == 2 and r['pp'] == 1 and r['eager'] and r['tensor_device'] == f'cuda:{rank}'
            assert r['cuda_visible_devices'] == '2,3'
            a = r['actual']; position = 192 + k - 2
            assert r['prompt_len'] == 192 and r['output_len_before'] == k - 1
            assert a['position'] == a['context_len'] == [position] and r['computed_before'] == position
            assert a['sequence_len'] == [position + 1] and r['sequence_len'] == position + 1
            assert a['input_token'] == [tokens[rid][k - 2]] and r['last_input_token'] == tokens[rid][k - 2]
            assert r['block_size'] == 16 and a['block_table'] == r['driver_block_table']
            assert a['slot'] == [a['block_table'][position // 16] * 16 + position % 16]
            assert r['ordered_request_seq_ids'][r['input_sequence_row']] == [rid, r['seq_id']]
            assert r['expected_query_len'] == 1 and a['query_start_end'] == [r['input_token_offset'], r['input_token_offset'] + 1]
            batch = 1 if rid == ids['golden-second'] else 2
            assert r['num_prefills'] == r['num_prefill_tokens'] == 0 and r['num_decode_tokens'] == batch
            step = steps[rid][k - 1]
            assert (step['prefill'], step['decode'], step['tokens']) == (0, batch, batch)
            assert step['request_ids'] == [x[0] for x in r['ordered_request_seq_ids']]
            assert step['started_s'] <= r['snapshot_enqueued_s'] <= r['recorded_s'] <= step['finished_s']
            if rank == 0:
                assert r['raw_logits_shape'] == [batch, 152064] and r['raw_logits_dtype'] == 'torch.bfloat16'
                assert r['logits_row'] == batch - 1 and r['sampler_parent_seq_id'] == r['seq_id']
                assert r['raw_model_logits_pre_sampler']['argmax_token_id'] == r['sampler_output_token'] == tokens[rid][k - 1]
            else:
                assert r['raw_model_logits_pre_sampler'] is None and r['sampler_output_token'] is None
            indexed[key] = dict(raw=r, source=str(path), line=line)
        assert set(indexed) == expected; ranks[rank] = indexed
    for key in expected:
        a, b = ranks[0][key]['raw'], ranks[1][key]['raw']
        for field in ('actual', 'ordered_request_seq_ids', 'seq_id', 'input_sequence_row', 'input_token_offset',
                'logits_row', 'computed_before', 'sequence_len', 'driver_block_table', 'block_size', 'last_input_token', 'output_len_before'):
            assert a[field] == b[field]
    return ranks


def main():
    native_spec, old_spec = read(NATIVE / 'observation-spec.json'), read(OLD / 'observation-spec.json')
    assert native_spec['requests'] == old_spec['requests'][:4]
    ids = {r['label']: r['request_uuid'] for r in native_spec['requests']}
    nt = read(NATIVE / 'results/child/full-outputs.json')['token_ids_by_request_uuid']
    ot = read(OLD / 'results/child/full-outputs.json')['token_ids_by_request_uuid']
    assert set(nt) == set(ids.values()) and len(nt) == 4
    assert all(len(v) == 64 and all(type(t) is int and t >= 0 for t in v) for v in nt.values())
    http = [r for r in lines(NATIVE / 'results/child/checks/http.jsonl') if r['route'] == '/native-reference']
    assert len(http) == 1 and http[0]['status'] == 200 and http[0]['body']['requests'] == native_spec['requests']
    assert http[0]['response']['complete'] and http[0]['response']['token_ids_by_request_uuid'] == nt
    log = lines(NATIVE / 'results/diagnostic-owner.events.jsonl')
    owner = [e for e in log if e['kind'] == 'executed_step']
    old_owner = lines(OLD / 'results/diagnostic-owner.events.jsonl')
    shape = lambda e: (e['prefill'], e['decode'], e['tokens'], e['request_ids'])
    old_nonempty = [e for e in old_owner if e['request_ids']]
    assert len(owner) == 197 and [shape(e) for e in owner] == [shape(e) for e in old_nonempty[:197]]
    assert len(owner[128:]) == 69 and all(not e['preempted'] and not e['blocks_to_swap_in']
        and not e['blocks_to_swap_out'] and not e['blocks_to_copy'] for e in owner)
    outputs = {rid: [] for rid in nt}; current = None; per_step = set(); emitted = 0
    for e in log:
        if e['kind'] == 'executed_step':
            if current is not None: assert per_step == set(current['request_ids'])
            current = e; per_step = set()
        elif e['kind'] == 'output':
            rid = e['request_id']; assert rid in current['request_ids'] and rid not in per_step
            assert e['token_ids'][:-1] == outputs[rid] and len(e['token_ids']) == len(outputs[rid]) + 1
            outputs[rid] = e['token_ids']; per_step.add(rid); emitted += 1
            assert e['finished'] is (len(outputs[rid]) == 64)
    assert outputs == nt and emitted == 256 and per_step == set(current['request_ids'])
    ni, oi = capture(NATIVE, nt, ids, owner), capture(OLD, ot, ids, old_owner)
    comparisons = []; rows = []
    for scope, left, right, lt, rt in [
        ('native-solo-vs-native-pair', ni, ni, nt, nt),
        ('old-solo-vs-native-solo', oi, ni, ot, nt),
        ('old-pair-vs-native-pair', oi, ni, ot, nt)]:
        label1 = 'golden-second' if 'solo' in scope else 'temporal-second'
        label2 = 'temporal-second' if scope == 'native-solo-vs-native-pair' else label1
        l_id, r_id = ids[label1], ids[label2]
        comparison = dict(scope=scope, first_difference=diff(lt[l_id], rt[r_id]), steps=[])
        for k in range(28, 35):
            l, r = left[0][l_id, k], right[0][r_id, k]
            common = lt[l_id][:k - 1] == rt[r_id][:k - 1]
            comparison['steps'].append(dict(output_index=k, complete_prior_output_prefix_equal=common,
                metadata_equal=l['raw']['actual'] == r['raw']['actual'], left_source=l['source'], left_line=l['line'],
                right_source=r['source'], right_line=r['line']))
        comparisons.append(comparison)
    for run, index in [('old003', oi), ('native', ni)]:
        for label in ('golden-second', 'temporal-second'):
            rid = ids[label]
            for k in range(28, 35):
                observed = index[0][rid, k]; r = observed['raw']; a = r['actual']; logits = r['raw_model_logits_pre_sampler']
                rows.append(dict(run=run,label=label,uuid=rid,output_index=k,input_token=r['last_input_token'],
                    position=a['position'][0],context_len=a['context_len'][0],sequence_len=r['sequence_len'],
                    sid=r['seq_id'],input_row=r['input_sequence_row'],sample_indices_row=r['logits_row'],
                    slot=a['slot'][0],block_table=a['block_table'],logit2776=logits['token_2776'],logit4172=logits['token_4172'],
                    delta4172minus2776=logits['token_4172']-logits['token_2776'],top2_ids=logits['top2_token_ids'],
                    top2_values=logits['top2_values'],argmax=logits['argmax_token_id'],output_token=r['sampler_output_token'],
                    source=observed['source'],line=observed['line']))
    parent_source = ROOT / 'campaign/B32B-temporal-solo-pair-capture-peer-review-v1/analyze.py'
    raw(parent_source); s=importlib.util.spec_from_file_location('independent_physical_reuse',parent_source)
    physical=importlib.util.module_from_spec(s);s.loader.exec_module(physical);physical.BASE=NATIVE;physical.FILES=FILES
    status = read(NATIVE / 'results/status.json'); measured = physical.physical_evidence(status)
    child = read(NATIVE / 'results/child/status.json')
    local_differences = {slot: diff(nt[ids['golden-'+slot]], nt[ids['temporal-'+slot]]) for slot in ('first', 'second')}
    assert child['first_differences'] == list(local_differences.values()) and status['exact_passed'] is False
    assert status['capture_complete'] and status['observation_completed']
    nlog=raw(NATIVE / 'results/diagnostic-docker.log').decode();olog=raw(OLD / 'results/diagnostic-docker.log').decode()
    config=lambda text: next(line.split('with config: ',1)[1] for line in text.splitlines() if 'Initializing a V0 LLM engine' in line)
    assert config(olog).replace('chunked_prefill_enabled=True','chunked_prefill_enabled=False') == config(nlog)
    code = ROOT / 'campaign/B32B-temporal-engine-review-v1/actual-sources'
    for path in (code/'worker/model_runner.py',code/'attention/backends/flash_attn.py',code/'pdblend_runtime.py'):
        raw(path)
    result=dict(schema=1,cpu_only=True,gpu_actions=False,actual_four_full64=256,owner_steps=197,pair_steps=69,
        full197_nonempty_owner_shapes_equal_old003=True,
        old003_empty_owner_events=[dict(line=i, **e) for i,e in enumerate(old_owner,1) if not e['request_ids']],
        native_empty_owner_steps=0,owner_outputs_equal_http=True,all28_native_rank_records_verified=True,
        all28_old_rank_records_reverified=True,native_vs_this_native_solo_first_differences=local_differences,
        old003_vs_native_first_differences={label:diff(ot[rid],nt[rid]) for label,rid in ids.items()},
        comparisons=comparisons,selected_records=rows,physical_evidence=measured,
        printed_engine_config_only_difference=dict(old_chunked_prefill_enabled=True,native_chunked_prefill_enabled=False),
        source_supported_unmeasured_branch_candidate=dict(builder_caches_initial_chunked_flag='worker/model_runner.py:509',
            prefill_block_table_construction='attention/backends/flash_attn.py:451',
            direct_KV_vs_paged_KV_prefill_branch='attention/backends/flash_attn.py:789',
            actual_prefill_branch_captured=False,causal_root_cause_proven=False),
        original_exact_failure_preserved=True,native_shape_aware_correctness_pass=False,ecoserve_eligible=False,
        inputs_sha256=FILES)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in FILES.items())
    result['input_sha256_stable_after']=True
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'four-way-selected-steps.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
    print(json.dumps(dict(native_differences=local_differences,old_vs_native=result['old003_vs_native_first_differences'],
        step32=[r for r in rows if r['output_index']==32],energy_j=measured['all8_integrated_energy_j'],input_files=len(FILES)),indent=2))


if __name__ == '__main__': main()
