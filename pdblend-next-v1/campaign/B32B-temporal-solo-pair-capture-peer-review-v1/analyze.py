"""Independent raw interpretation; no inference, hardware, or upstream edits."""
import collections
import csv
import hashlib
import json
import io
import math
from pathlib import Path

ROOT = Path('/root/workspace/pdblend-next-v1')
BASE = ROOT/'campaign/B32B-temporal-observation-attempt-003'
OUT = Path(__file__).resolve().parent
FILES = {}


def raw(p):
    p = Path(p).resolve()
    data = p.read_bytes()
    FILES[str(p)] = hashlib.sha256(data).hexdigest()
    return data


def read(p):
    return json.loads(raw(p))


def lines(p):
    return [json.loads(x) for x in raw(p).splitlines()]


def diff(a,b):
    assert len(a) == len(b)
    return next((dict(position_one_based=i, reference=x, observed=y)
                 for i,(x,y) in enumerate(zip(a,b),1) if x!=y), None)


def same_prefix_comparison(tokens, ids, records):
    solo, temporal = ids['golden-second'], ids['temporal-second']
    first = diff(tokens[solo], tokens[temporal])
    table = {(r['request_uuid'], r['output_index']): r for r in records}
    result = []
    for k in range(28, 35):
        a, b = table[solo, k], table[temporal, k]
        common = tokens[solo][:k-1] == tokens[temporal][:k-1]
        result.append(dict(output_index=k, all_output_prefix_before_equal=common,
            same_input_token=a['input_token']==b['input_token'],
            included_in_same_prefix_comparison=common,
            solo={x:a[x] for x in ('input_token','position','sequence_len','top2_ids','top2_values',
                'logit_2776','logit_4172','logit_delta_4172_minus_2776','argmax','output_token',
                'sequence_id','input_row','logits_row_from_sample_indices','slot','block_table')},
            temporal={x:b[x] for x in ('input_token','position','sequence_len','top2_ids','top2_values',
                'logit_2776','logit_4172','logit_delta_4172_minus_2776','argmax','output_token',
                'sequence_id','input_row','logits_row_from_sample_indices','slot','block_table')}))
    return dict(first_difference=first, same_prefix_steps=[r['output_index'] for r in result
        if r['included_in_same_prefix_comparison']], steps=result)


def physical_evidence(status):
    # Independent full-window all-eight trapezoids, clipped by linear endpoints.
    start, end = status['operation_start_s'], status['operation_end_s']
    power = list(csv.DictReader(io.StringIO(raw(BASE/'results/power/power.csv').decode())))
    points = [(float(r['t_s']), [float(r[f'gpu{i}_w']) for i in range(8)]) for r in power]
    assert points[0][0] <= start < end <= points[-1][0]
    assert all(math.isfinite(v) and v >= 0 for _, values in points for v in values)
    energy = [0.] * 8
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        assert t1 > t0
        a, b = max(t0, start), min(t1, end)
        if b <= a:
            continue
        for i in range(8):
            at = p0[i] + (p1[i]-p0[i])*(a-t0)/(t1-t0)
            bt = p0[i] + (p1[i]-p0[i])*(b-t0)/(t1-t0)
            energy[i] += (at+bt)*.5*(b-a)
    total = sum(energy)
    assert abs(total-status['full_operation_energy_j']) < 1e-6
    assert status['complete'] and status['measurement_valid'] and not status['errors']
    assert status['all_original_restored'] and status['clock_restore_complete']
    assert all(status['process_terminal'].values())
    assert status['power_evidence']['power_mode']=='instant'
    assert status['power_evidence']['power_source_verified']
    native = {}
    for label, expected in [('parent-diagnostic-native',1),('restored-native',4)]:
        proof = read(BASE/'results'/label/'checks.json')['cleanup']
        key = 'diagnostic_native' if expected == 1 else 'restored_native'
        assert proof == status[key] and proof['complete'] and not proof['errors']
        assert len(proof['instances']) == expected
        for iid, r in proof['instances'].items():
            assert not r['errors']
            p = r['proof']
            assert p['drained'] and p['drain_proof_type']=='synchronous_put_owner_barrier'
            assert len(p['transfers']) == 2
            for rank in p['transfers']:
                assert all(type(rank[k]) is int and rank[k]==0 for k in
                    ('buffered_tensors','inflight_receives','buffered_gpu_bytes'))
                assert rank['allocations']=={} and rank['listener_alive']
            after = r['after']
            assert after['generation']==after['acknowledged_generation']
            assert after['accepting'] and after['admit_prefill'] and after['admit_decode']
            assert after['mode']=='continuous' and after['role']=='mixed'
            assert all(after[k]==0 for k in ('active','running','waiting'))
            assert after['kv_allocations']=={} and after['transfer_allocations']=={}
            assert after['runtime_error'] is None and after['error'] is None
            native[iid] = dict(generation=after['generation'], ranks=2, accepting=True,
                mode=after['mode'], send_counters_observed=after['transfer_inflight_sends_observed'])
    before = read(BASE/'results/original-identity.before.json')
    after = read(BASE/'results/restored-identity.after.json')
    assert isinstance(before, list) and isinstance(after, list) and len(before)==len(after)==4
    after = {x['container']['Id']:x for x in after}
    restored = []
    for wrapper in before:
        a = wrapper['container']
        fresh = after[a['Id']]
        b = fresh['container']
        assert wrapper['provenance'] == fresh['provenance']
        for key in ('Id','Name','Image','Path','Args','Config','HostConfig'):
            assert a[key] == b[key], key
        assert sorted(a['Mounts'], key=lambda x:json.dumps(x,sort_keys=True)) == sorted(b['Mounts'], key=lambda x:json.dumps(x,sort_keys=True))
        assert b['State']['Running'] and b['State']['StartedAt']!=a['State']['StartedAt']
        assert b['State']['Pid']>0 and b['State']['Pid']!=a['State']['Pid']
        restored.append(dict(container_id=a['Id'], old_host_pid=a['State']['Pid'],
            new_host_pid=b['State']['Pid'], old_started_at=a['State']['StartedAt'],
            new_started_at=b['State']['StartedAt']))
    bootstrap = read(BASE/'results/restored-bootstrap.binding.json')
    assert status['restored_binding_sha256'] == FILES[str(BASE/'results/restored-bootstrap.binding.json')]
    assert bootstrap['configs'] == {} and bootstrap['output_correctness_verified'] is False
    assert bootstrap['correctness_gate_required_before_performance'] is True
    for instance in bootstrap['instances']:
        actual = after[instance['container']['id']]
        assert instance['container']['StartedAt'] == actual['container']['State']['StartedAt']
        assert instance['container']['image'] == actual['container']['Image']
        assert instance['provenance'] == actual['provenance']
    clocks = list(csv.DictReader(io.StringIO(raw(BASE/'results/power/clocks.csv').decode())))
    assert clocks and all(f'gpu{i}_sm_mhz' in clocks[0] for i in range(8))
    clock_ranges={str(i): [min(float(r[f'gpu{i}_sm_mhz']) for r in clocks),
        max(float(r[f'gpu{i}_sm_mhz']) for r in clocks)] for i in range(8)}
    return dict(full_operation_start_s=start, full_operation_end_s=end,
        full_operation_duration_s=end-start, all8_integrated_energy_j=total,
        per_gpu_energy_j=energy, reported_energy_j=status['full_operation_energy_j'],
        recomputed_minus_reported_j=total-status['full_operation_energy_j'],
        power_samples=len(points),clock_samples=len(clocks),actual_clock_ranges_mhz=clock_ranges,
        native_restore=native, same_container_identity_fresh_process=restored,
        legacy_unknown_send_counters_not_fabricated=True,
        performance_evidence=False, does_not_sum_overlapping_child_or_attempt_windows=True)


def main():
    spec = read(BASE/'observation-spec.json')
    prepared = read(BASE/'spec.json')
    assert prepared['observation_spec_sha256'] == FILES[str(BASE/'observation-spec.json')]
    for relative, installed in [('pdblend_diagnostics.py','vllm/pdblend_diagnostics.py'),
            ('model_runner.py','vllm/worker/model_runner.py')]:
        p = BASE/'image-context'/relative
        raw(p)
        assert prepared['installed_diagnostic_sources'][
            '/usr/local/lib/python3.10/dist-packages/'+installed] == FILES[str(p)]
    jobs = read(BASE/'results/job.json')
    binding = read(BASE/'results/diagnostic-binding.json')
    assert jobs['binding_sha256'] == FILES[str(BASE/'results/diagnostic-binding.json')]
    assert jobs['binding'] == binding
    assert jobs['observation_spec_sha256'] == prepared['observation_spec_sha256']
    assert binding['instances'][0]['gpus']==[2,3] and binding['instances'][0]['tp']==2
    assert binding['instances'][0]['provenance']['cuda_visible_devices']=='2,3'
    output = read(BASE/'results/child/full-outputs.json')
    tokens = output['token_ids_by_request_uuid']
    ids = {r['label']:r['request_uuid'] for r in spec['requests']}
    assert len(ids) == len(set(ids.values())) == 6 and set(tokens) == set(ids.values())
    assert all(len(t)==64 and all(type(n) is int for n in t) for t in tokens.values())
    assert spec['request_ids'] == [ids['golden-second'],ids['temporal-second']]
    http = lines(BASE/'results/child/checks/http.jsonl')
    completed = [r for r in http if r['route']=='/v1/completions']
    assert len(completed)==6 and {r['request_id'] for r in completed}==set(ids.values())
    by_id = {r['request_uuid']:r for r in spec['requests']}
    for r in completed:
        rid = r['request_id']
        assert r['body']==by_id[rid]['body'] and r['status']==200
        assert r['response']['token_ids']==tokens[rid]
        assert r['response']['usage']['completion_tokens']==64
        assert r['response']['usage']['prompt_tokens']==by_id[rid]['prompt_length']
    child_status = read(BASE/'results/child/status.json')
    assert child_status['complete'] and child_status['completed_requests']==6
    assert child_status['cleanup_complete'] and not child_status['errors']
    assert child_status['job_sha256']==FILES[str(BASE/'results/job.json')]
    owner = lines(BASE/'results/diagnostic-owner.events.jsonl')
    by_request = {rid:[dict(event=e, line=i) for i,e in enumerate(owner,1)
                        if rid in e['request_ids']] for rid in ids.values()}
    assert all(len(e)==64 for e in by_request.values())
    expected = {(rid,k) for rid in spec['request_ids'] for k in range(28,35)}
    ranks = {}
    for rank in (0,1):
        paths = list((BASE/'results/capture-frozen').glob(f'rank{rank}-pid*.jsonl'))
        assert len(paths)==1
        path = paths[0]
        values = lines(path)
        status = read(path.with_suffix('.status.json'))
        for original in (path,path.with_suffix('.status.json')):
            assert raw(BASE/'results/capture-live'/original.name) == original.read_bytes()
        assert status['complete'] and not status['failed'] and status['error'] is None
        assert status['written']==status['expected_records']==len(values)==14
        current = {}
        for line,r in enumerate(values,1):
            key = r['request_id'],r['output_index']
            assert key not in current and key in expected
            assert r['rank']==rank==status['rank'] and r['pid']==status['pid']
            assert r['spec_sha256']==prepared['observation_spec_sha256'] and r['observer_error'] is None
            assert r['tp']==2 and r['pp']==1 and r['eager'] and r['tensor_device']==f'cuda:{rank}'
            assert r['cuda_visible_devices']=='2,3'
            rid,k = key
            a = r['actual']
            n = 192
            assert r['prompt_len']==n and r['output_len_before']==k-1
            assert a['position']==[n+k-2] and r['computed_before']==n+k-2
            assert a['sequence_len']==[n+k-1] and r['sequence_len']==n+k-1
            assert a['context_len']==[n+k-2]
            assert a['input_token']==[tokens[rid][k-2]] and r['last_input_token']==tokens[rid][k-2]
            assert r['block_size']==16 and a['block_table']==r['driver_block_table']
            position = a['position'][0]
            assert a['slot']==[a['block_table'][position//16]*16+position%16]
            assert r['ordered_request_seq_ids'][r['input_sequence_row']]==[rid,r['seq_id']]
            assert r['expected_query_len']==1 and a['query_start_end']==[r['input_token_offset'],r['input_token_offset']+1]
            expected_batch = 1 if rid == ids['golden-second'] else 2
            assert r['num_prefills']==r['num_prefill_tokens']==0 and r['num_decode_tokens']==expected_batch
            step = by_request[rid][k-1]['event']
            assert step['prefill']==0 and step['decode']==step['tokens']==expected_batch
            assert step['request_ids']==[x[0] for x in r['ordered_request_seq_ids']]
            assert step['started_s'] <= r['snapshot_enqueued_s'] <= r['recorded_s'] <= step['finished_s']
            if rank==0:
                assert r['raw_logits_shape']==[expected_batch,152064] and r['raw_logits_dtype']=='torch.bfloat16'
                assert r['logits_row']==expected_batch-1 and r['sampler_parent_seq_id']==r['seq_id']
                assert r['sampler_output_token']==tokens[rid][k-1]
                assert r['raw_model_logits_pre_sampler']['argmax_token_id']==r['sampler_output_token']
            else:
                assert r['raw_model_logits_pre_sampler'] is None and r['sampler_output_token'] is None
            current[key] = dict(record=r,source=str(path),line=line)
        assert set(current)==expected
        ranks[rank]=current
    table=[]
    for key in sorted(expected):
        a,b=(ranks[k][key]['record'] for k in (0,1))
        assert a['pid']!=b['pid']
        equal_fields=('actual','ordered_request_seq_ids','seq_id','input_sequence_row',
                      'input_token_offset','logits_row','computed_before','sequence_len',
                      'driver_block_table','block_size','last_input_token','output_len_before')
        assert all(a[f]==b[f] for f in equal_fields)
        rid,k=key
        logits=a['raw_model_logits_pre_sampler']
        e=by_request[rid][k-1]
        table.append(dict(phase='temporal' if rid==ids['temporal-second'] else 'solo',
            request_uuid=rid,output_index=k,sequence_id=a['seq_id'],input_row=a['input_sequence_row'],
            logits_row_from_sample_indices=a['logits_row'],position=a['actual']['position'][0],
            sequence_len=a['sequence_len'],block_table=a['actual']['block_table'],
            slot=a['actual']['slot'][0],input_token=a['last_input_token'],output_token=a['sampler_output_token'],
            rank_metadata_equal=True,logical_formulas_and_http_equal=True,
            top2_ids=logits['top2_token_ids'],top2_values=logits['top2_values'],
            logit_2776=logits['token_2776'],logit_4172=logits['token_4172'],
            logit_delta_4172_minus_2776=logits['token_4172']-logits['token_2776'],
            argmax=logits['argmax_token_id'],owner_line=e['line'],owner_event=e['event'],
            rank0_source=ranks[0][key]['source'],rank0_line=ranks[0][key]['line'],
            rank1_source=ranks[1][key]['source'],rank1_line=ranks[1][key]['line']))
    phases={}
    for phase in ('temporal','continuous'):
        first,second=ids[phase+'-first'],ids[phase+'-second']
        initial=by_request[second][0]
        phases[phase]=dict(first_difference=[diff(tokens[ids['golden-'+slot]],tokens[ids[phase+'-'+slot]])
            for slot in ('first','second')],second_prefill=initial,
            first_outputs_before_second_prefill=sum(first in e['request_ids'] for e in owner[:initial['line']-1]),
            first_in_second_prefill=first in initial['event']['request_ids'],
            second_request_step_shapes=dict(collections.Counter(
                str((x['event']['prefill'],x['event']['decode'])) for x in by_request[second])))
    prior = ROOT/'campaign/B32B-temporal-observation-attempt-001/results/child/full-outputs.json'
    previous = read(prior)['token_ids_by_request_uuid']
    original_status=read(BASE/'results/status.json')
    physical = physical_evidence(original_status)
    prefix = same_prefix_comparison(tokens, ids, table)
    second_prior = read(ROOT/'campaign/B32B-temporal-observation-attempt-002/results/child/full-outputs.json')['token_ids_by_request_uuid']
    derived_exact = all(v['first_difference']==[None,None] for v in phases.values())
    assert original_status['exact_passed']==child_status['exact_passed']==derived_exact
    result=dict(schema=1,read_only=True,gpu_executed_by_this_audit=False,actual_capture_complete=True,
        capture_live_and_frozen_original_bytes_equal=True,
        metadata_checks_all_passed=True,exact_gate_passed=derived_exact,kv_contents_verified=False,
        actual_uuid_by_label=ids,all_six_full64=True,per_request_owner_steps={k:len(v) for k,v in by_request.items()},
        owner_total_events=len(owner),phases=phases,selected_records=table,
        all_six_outputs_equal_attempt001_disabled_hook=previous==tokens,
        all_six_outputs_equal_attempt002_pair_hook=second_prior==tokens,
        same_prefix_comparison=prefix,physical_evidence=physical,
        capture_host_readback_max_s=max(x['record']['host_readback_s'] for rank in ranks.values() for x in rank.values()),
        rank0_logits_enqueue_max_s=max(x['record']['logits_enqueue_host_s'] for x in ranks[0].values()),
        original_status_unchanged=dict(capture_complete=original_status['capture_complete'],
            capture_error=original_status.get('capture_error'),measurement_valid=original_status['measurement_valid'],
            all_original_restored=original_status['all_original_restored'],
            reported_full_operation_energy_j=original_status['full_operation_energy_j']),
        caveats=['Complete two-rank metadata is not KV-content proof.',
            'logits_row was derived from real SamplingMetadata.sample_indices by the bound hook; full sampling metadata was not dumped.',
            'Only steps whose complete prompt/output prefix matches are used for solo versus temporal same-input comparison.',
            'Same input token alone after divergence is not a common-prefix comparison.',
            'An argmax/top2 tie need not use the same index order; top-k ordering alone is not an output oracle.',
            'Identical six outputs across two attempts does not prove observation timing has no effect.'],
        input_sha256=FILES)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in FILES.items())
    result['all_input_sha256_unchanged_after']=True
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    columns=[k for k in table[0] if k!='owner_event']
    with (OUT/'selected-steps.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in table:writer.writerow({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in columns})
    print(json.dumps(dict(records=len(table),two_rank_records=28,all_metadata_checks=True,
                         six_outputs_equal_prior=previous==tokens,input_files=len(FILES),
                         prefix=prefix,physical_evidence=physical),indent=2))


if __name__=='__main__':main()
