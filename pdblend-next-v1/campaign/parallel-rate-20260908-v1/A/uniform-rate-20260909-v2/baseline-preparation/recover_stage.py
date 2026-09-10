"""Complete only the failed cache interface stage using independently composed proof."""
import argparse,fcntl,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;U=HERE.parent;ROOT=U.parents[1]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
sys.path.insert(0,str(HERE/'qualification-v2'))
from qualify import child

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--definition',type=Path,required=True);a=ap.parse_args()
    d=p.checked(p.ref(a.definition))
    for item in d['sources']:p.need(p.sha(item['path'])==item['sha256'],'recovery stage source changed')
    previous=p.checked(d['predecessor_status']);retention=p.checked(d['retention_status'])
    p.need(previous.get('finished_s') and not previous['node_lease_held'] and not p.active_owner(previous),'old failed stage must be stopped')
    p.need(retention['passed'] and retention['complete'] and retention.get('finished_s') and not retention['node_lease_held'],'actual cache supplement incomplete')
    out=Path(d['out']);p.need(not out.exists(),'fresh recovery output required')
    lock=(U/'baseline-stage.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True);state=dict(schema='uniform-v2-baseline-recovery-stage-status',node='Anew20260909',model='14b',
        pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],started_s=time.time(),
        node_lease_held=False,complete=False,phase='compose',definition=p.ref(a.definition));p.save(out/'status.json',state)
    try:
        qout=out/'qualification';qsource=HERE/'recovery-qualification-v1'
        child([sys.executable,'-B',str(qsource/'qualify.py'),'--bootstrap',d['bootstrap']['path'],
            '--previous-qualification',d['previous_qualification'],'--retention-out',d['retention_out'],
            '--out',str(qout),'--node','Anew20260909','--hostname','iZwz9274emxme9019d2sjgZ',
            '--profile',d['profile']['path'],'--run'],out,state)
        bindings=p.read(qout/'bindings.json')
        for system,reference in bindings.items():
            state['phase']='cache-'+system;p.save(out/'status.json',state);cache_out=out/'qualification-caches'/system
            child([sys.executable,'-B',str(HERE/'optional_cache_v4.py'),'--helper',d['cache_helper']['path'],
                '--qualification',reference['path'],'--validator',str(qsource/'verify.py'),'--out',str(cache_out)],out,state)
            selection=p.read(cache_out/'selection.json');p.need(selection['qualification']==reference and not selection['interrupted'],'cache selection mismatch')
            state.setdefault('qualification_caches',{})[system]=p.ref(cache_out/'selection.json')
            state['phase']='collector-'+system;p.save(out/'status.json',state);destination=out/'formal'/system
            child([sys.executable,'-B',d['meter_binding']['path'],'--binding',reference['path'],
                '--native-validator',selection['qualification_validator']['path'],'--out',str(destination)],out,state)
            qref=p.ref(destination/'qualified.json')
            handoff=dict(node='Anew20260909',model='14b',system=system,qualification=qref,
                qualification_validator=d['meter_binding'],predecessors=[p.ref(qout/'status.json')],extra_files=[p.ref(a.definition)])
            for dataset in p.checked(reference)['configs']:p.save(out/'handoffs'/(dataset+'-'+system+'.json'),handoff)
        actual={}
        for key,path in d['handoffs'].items():
            dataset,system=key.split(':');h=p.checked(p.ref(path));q=p.checked(h['qualification']);b=p.checked(q['binding'])
            p.need(h['system']==b['system']==system and dataset in b['configs'],'missing actual dataset/system handoff')
            config=p.read(b['configs'][dataset]);p.need(config['comparison_system']==system and config['max_service_frequency_mhz']==2100,'actual recovered platform/policy changed')
            actual[key]=dict(handoff=p.ref(path),binding=q['binding'],configuration=p.ref(b['configs'][dataset]))
        p.need(set(actual)=={x+':'+y for x,y in d['groups']},'recovery handoff coverage incomplete')
        p.save(out/'handoff-matrix-verification.json',dict(passed=True,groups=actual,all_declared_groups_present=True))
        state.update(complete=True,phase='qualified',bindings=p.ref(qout/'bindings.json'))
    except BaseException as exc:state.update(phase='stopped_failure',error=repr(exc));raise
    finally:state.update(finished_s=time.time(),node_lease_held=False);p.save(out/'status.json',state);lock.close()
if __name__=='__main__':main()
