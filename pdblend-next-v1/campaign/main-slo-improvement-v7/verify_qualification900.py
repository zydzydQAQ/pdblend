"""Audit actual paired development cycles, including failed functionality."""
import argparse
import json
from pathlib import Path
import sys
import time
import protocol as p


def actual_source(spec, directory, result):
    p.need(p.read(directory/'spec-reference.json')==p.ref(spec),
           'actual 900s invocation used another frozen specification')
    declared=p.read(spec)
    expected=dict(original_binding=declared['original_binding'],capacity_binding=declared['capacity_binding'],
        config=declared['config'],host_manifest=p.ref(Path(declared['host_release'])/'manifest.json'))
    p.need(result.get('source')==expected,'900s result source differs from the declared arm')
    p.need(result.get('artifacts') and all(p.sha(path)==digest for path,digest in result['artifacts'].items()),
           '900s request/operation artifacts changed')
    requests=directory/'qualification900/requests.json'
    p.need(result['artifacts'].get(str(requests.resolve()))==p.sha(requests),
           '900s request rows are not the retained measured artifact')
    p.need(all(p.sha(path)==digest for path,digest in declared['files'].items()),
           '900s declared implementation changed')


def retained_identity(directory, binding):
    before=p.read(directory/'identity.before.json');after=p.read(directory/'identity.after.json')
    expected={i['id']:i for i in binding['instances']}
    p.need(len(before)==len(after)==len(expected)==2, 'original two native identity proofs required')
    for rows in (before,after):
        p.need({r['provenance']['instance_id'] for r in rows}==set(expected),'missing or foreign original owner')
        for row in rows:
            instance=expected[row['provenance']['instance_id']];container=row['container']
            p.need(container['Id']==instance['container']['id']
                   and container['Image']==instance['container']['image']
                   and container['State']['StartedAt']==instance['container']['StartedAt']
                   and container['State']['Running'] is True,
                   'retained engine was restarted or changed during 900s')
            p.need(all(row['provenance'].get(k)==v for k,v in instance['provenance'].items()),
                   'retained engine actual imported source/model changed')
    p.need({r['container']['Id']:r['container']['State']['Pid'] for r in before}==
           {r['container']['Id']:r['container']['State']['Pid'] for r in after},
           'retained engine PID changed during 900s')


def verify(pair_path, fixed_out, dynamic_out, code, out):
    p.need(not out.exists(), 'new qualification receipt required')
    pair=p.read(pair_path);trace=p.checked(pair['trace'])
    sys.path.insert(0,str(code.resolve()))
    from capacity_certificate import raw_measurement
    report=dict(schema='capacity-900-pair-qualification-v1',created_s=time.time(),passed=False,
        pair=p.ref(pair_path),formal_eligible=False,independent_repetitions_per_arm=1,
        arms={},performance_thresholds_not_waived=True)
    for arm,directory in [('fixed2',fixed_out),('dynamic',dynamic_out)]:
        packet=dict(expected_output=str(directory.resolve()),verified=False)
        for key,name in [('result','qualification900/result.json'),('status','status.json')]:
            if (directory/name).is_file():packet[key]=p.ref(directory/name)
        report['arms'][arm]=packet
    try:
        for arm,directory in [('fixed2',fixed_out),('dynamic',dynamic_out)]:
            spec=p.checked(pair['arms'][arm]);status=p.read(directory/'status.json')
            result=p.read(directory/'qualification900/result.json')
            report['arms'][arm].update(result=p.ref(directory/'qualification900/result.json'),
                status=p.ref(directory/'status.json'),raw_measurement=result.get('raw_measurement'),
                reported_energy_j=result.get('energy_j'),reported_slo_attainment=result.get('slo_attainment'))
            actual_source(Path(pair['arms'][arm]['path']),directory,result)
            p.need(status.get('complete') is True and status.get('cleanup_complete') is True
                   and not status.get('error') and not status.get('cleanup_errors'),arm+' did not cleanly finish')
            p.need(result.get('trace')==pair['trace'] and result.get('complete') is True
                   and result.get('work_complete') is True and result.get('native_idle') is True,
                   arm+' incomplete prescribed development work')
            config=p.checked(spec['config']);rows=p.read(directory/'qualification900/requests.json')
            retained_identity(directory,p.checked(spec['original_binding']))
            n=trace['n_requests']
            p.need(len(rows)==len(trace['requests'])==n and len({r['request_id'] for r in rows})==n,
                   arm+' request cardinality changed')
            good=0
            for row,request in zip(rows,trace['requests']):
                p.need(row.get('success')==1 and row.get('token_ids_verified')==1
                       and row['generated_tokens']==row['output_len']==request['output_len']
                       and row['input_tokens']==row['prompt_len']==request['prompt_len'],
                       arm+' did not complete original token workload')
                ok=(row.get('ttft_s') is not None and row.get('tpot_s') is not None
                    and row['ttft_s']<config['slo_ttft_s'] and row['tpot_s']<config['slo_tpot_s'])
                p.need(row['slo_ok']==int(ok),arm+' SLO differs from raw latency')
                good+=ok
            raw=raw_measurement(result['raw_measurement'])
            p.need(result['n_expected']==result['n_rows']==n and result['n_good']==good
                   and abs(result['slo_attainment']-good/n)<1e-12
                   and abs(result['energy_j']-raw['energy_j'])<=max(1e-7,raw['energy_j']*1e-9),
                   arm+' reported denominator/SLO/energy differs from retained raw evidence')
            epoch=result['actual_arrival_epoch_s']
            p.need(raw['measurement_start_s']<=epoch and raw['measurement_end_s']>=epoch+900,
                   arm+' energy window truncates the declared cycle')
            commits=[]
            if arm=='dynamic':
                inventory=p.read(directory/'inventory.json')
                p.need(inventory['complete'] is True and inventory['transition_inflight'] is False
                       and inventory['identity']==p.checked(spec['capacity_binding'])['identity']
                       and len(inventory['initial_ids'])==2
                       and set(inventory['initial_ids'])=={i['id'] for i in p.checked(spec['original_binding'])['instances']}
                       and {i['id'] for i in inventory['active_instances']}==set(inventory['initial_ids']),
                       'dynamic final physical layout not original two')
                p.need(not any(e['kind']=='transition_failed' for e in inventory['events']),
                       'actual dynamic transaction failure retained')
                commits=[e for e in inventory['events'] if e['kind']=='physical_commit'
                    and e.get('scope')=='policy_transition' and e.get('execution_verified') is True]
                adds=[e for e in commits if e['operation']=='restore' and epoch<=e['started_s']<epoch+600
                    and len(e['live_instances'])==3]
                removes=[e for e in commits if e['operation']=='remove' and epoch+600<=e['started_s']
                    and e['finished_s']<=epoch+900 and len(e['live_instances'])==2]
                p.need(adds and removes and any(a['instance_id']==r['instance_id']
                       and a['finished_s']<r['started_s'] for a in adds for r in removes),
                       'actual policy-driven add then low-phase remove was not observed before final cleanup')
            report['arms'][arm]=dict(result=p.ref(directory/'qualification900/result.json'),
                status=p.ref(directory/'status.json'),raw_measurement=result['raw_measurement'],
                n_requests=n,n_good=good,slo_attainment=good/n,energy_j=raw['energy_j'],
                measurement_duration_s=raw['measurement_end_s']-raw['measurement_start_s'],
                policy_transitions=commits,full_work=True,verified=True)
        f,d=report['arms']['fixed2'],report['arms']['dynamic']
        report.update(passed=True,actual_add_and_remove_within_trace=True,
            energy_change_pct=100*(d['energy_j']/f['energy_j']-1),
            slo_difference_pp=100*(d['slo_attainment']-f['slo_attainment']),
            energy_no_worse_than_fixed=d['energy_j']<=f['energy_j'],
            slo_at_least_90_percent=d['slo_attainment']>=.9,
            performance_passed=d['energy_j']<=f['energy_j'] and d['slo_attainment']>=min(.9,f['slo_attainment']))
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        p.write(out,report,exclusive=True)
    return p.ref(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','fixed-out','dynamic-out','code','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.pair,args.fixed_out,args.dynamic_out,args.code,args.out)))
