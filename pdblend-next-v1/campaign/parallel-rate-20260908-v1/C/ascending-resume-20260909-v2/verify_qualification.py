"""Replay saved C TP1 frequency/cancellation evidence without GPU or networking."""
from pathlib import Path
import cold_restore as c
import qualify as q


def verify(reference):
    qualification=c.checked(reference)
    assert qualification['schema']=='C7B-saved-frequency-legacy-qualification-v1'
    for group in ('files','source_files'):
        for path,digest in qualification[group].items():
            assert c.sha(path)==digest,path
    spec=c.checked(qualification['spec'])
    binding,ordinary=q.validate(spec,qualification['cold_restoration'])
    cold=c.checked(qualification['cold_restoration'])
    assert qualification['binding']==cold['binding'] and qualification['ordinary']==cold['ordinary']
    state=c.checked(qualification['status'])
    assert state['spec']==qualification['spec'] and state['cold_restoration']==qualification['cold_restoration']
    assert state['passed'] and not state.get('error') and not state['cleanup_errors']
    assert state['finished_s'] and not state['node_lease_held'] and state['clock_restore_complete']
    assert state['measurement']['measurement_valid'] and all(v['complete'] for v in state['restoration'].values())
    by_id={i['id']:i for i in binding['instances']}
    expected={r['instance_id']:r['response']['token_ids'] for r in ordinary['replies'] if r['prompt_length']==128}
    cases=state['frequency_cases']
    assert len(cases)==8 and {(x['instance_id'],x['frequency_mhz']) for x in cases}=={
        (i,f) for i in by_id for f in (900,1500,2100,2520)}
    clocks=c.read(Path(qualification['status']['path']).parent/'power/clocks.json')
    for case in cases:
        instance=by_id[case['instance_id']]
        for name in ('warmup','request'):
            row=case[name]
            assert row['success'] and row['http_status']==200 and row['done_marker'] and not row.get('error')
            assert row['prompt_token_ids']==([9707,1879,13]*43)[:128]
            assert row['output_token_ids']==expected[instance['id']]
            assert row['usage']['prompt_tokens']==128 and row['usage']['completion_tokens']==64
            assert len(row['token_received_s'])==len(row['output_token_ids'])==64
            assert row['token_received_s']==sorted(row['token_received_s'])
            tokens=[t for event in row['stream_events'] for t in event['event'].get('token_ids',[])]
            assert tokens==row['output_token_ids']
        row=case['request']
        assert q.clock_window(clocks,instance['gpus'],case['frequency_mhz'],
                              row['token_received_s'][0],row['token_received_s'][-1])==case['loaded_clock']
        q.legacy_idle(case['native_after'],instance)
    assert len(state['cancellations'])==2 and {x['instance_id'] for x in state['cancellations']}==set(by_id)
    for item in state['cancellations']:
        instance=by_id[item['instance_id']];e=item['evidence'];rid=e['request_id']
        assert e['verified'] and e['before']['active'] and e['before']['running']
        assert e['before']['kv_allocations'].get(rid,0)>0 and e['cancelled']['cancelled']==rid
        assert e['response']['status']>=400 and 'cancel' in e['response']['body'].lower()
        q.legacy_transfer_rows(e['cancelled']['transfers']);q.legacy_idle(e['settled'],instance)
    for instance in binding['instances']:
        restore=state['restoration'][instance['id']]
        proof=restore['proof'];before=restore['before']
        assert proof['drained'] and not proof['accepting'] and proof['generation']==before['generation']+1
        assert proof['drain_proof_type']=='synchronous_put_owner_barrier'
        q.legacy_transfer_rows(proof['transfers'])
        q.legacy_idle(restore['resumed']['after'],instance)
    return dict(passed=True,independently_recomputed=True,node='C',model='7b',
                binding=qualification['binding'],host_manifest=c.checked(spec['cold_spec'])['host_manifest'],
                qualification=reference,frequency_cases=8,controlled_legacy_cancellations=2,
                no_V3_sender_counter_claim=True)
