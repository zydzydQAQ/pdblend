"""Prepare immutable C baseline continuation using each historical controller source."""
import copy,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
base=REPO/'releases/five-system100-C7B-baseline-v1-runtime';coop=REPO/'releases/five-system100-C7B-baseline-cooperative-v1-runtime'
HOSTS={s:str(coop if s=='dynamollm' else base) for s in ('mixed','distserve','dynamollm','ecoserve')}
a,b=read(base/'manifest.json'),read(coop/'manifest.json')
changed=[k for k in set(a['files'])|set(b['files']) if a['files'].get(k)!=b['files'].get(k)]
assert changed==['src/ecopadg/serving/admission.py'];assert b['parent_manifest_sha256']==sha(base/'manifest.json')
old=(base/changed[0]).read_text();new=(coop/changed[0]).read_text()
extra='''            # Ready items and uncontended async locks need not suspend.
            # Yield before selection so every dispatcher branch remains cooperative.
            await asyncio.sleep(0)
'''
assert new.replace(extra,'')==old
# Never rewrite the existing v1 preparation or its declaration.
d=copy.deepcopy(read(HERE/'boundary-p4/declaration.json'));d.update(schema='parallel-rate-C-all-new-rate-pairs-p4-baseline-v2',created_s=time.time(),
 supersedes_unexecuted_baseline_declaration=str(HERE/'boundary-p4/declaration.json'),baseline_controller_hosts=HOSTS,
 source_selection_reason='Dynamo reuses the original cooperative controller that already produced 21 snapshot006 points; other baselines keep their original source.',
 baseline_scheduling_policy_profiles_budgets_unchanged=True)
for row in d['cells']:
 row['cell_id']=row['cell_id'].replace('parallel-rate-p4-explore-','parallel-rate-p4-baseline-v2-')
 row['baseline_controller_host']=HOSTS[row['system']]
for host in (base,coop):
 m=read(host/'manifest.json')
 for name,h in m['files'].items():assert sha(host/name)==h;d['files'][str(host/name)]=h
 d['files'][str(host/'manifest.json')]=sha(host/'manifest.json')
for p in [REPO/'campaign/cooperative-admission-yield-v1/README.md',REPO/'campaign/C7B-dynamo-cooperative-execution-v1/README.md']:
 d['files'][str(p)]=sha(p)
write(HERE/'boundary-p4v2/declaration.json',d)
# Lifecycle-only binder plus explicit per-system source guard; all old policies remain the original imported implementation.
newroot=HERE/'baseline-until-complete-v2';newroot.mkdir()
s=(HERE/'baseline-until-complete-v1/bind.py').read_text()
marker="    spec=read(a.spec);scope(spec);datasets=selected(spec,a.strategy,a.dataset)"
replace="""    spec=read(a.spec);scope(spec);datasets=selected(spec,a.strategy,a.dataset)
    system='mixed' if not a.strategy else 'dynamollm' if a.strategy.startswith('dynamollm') else a.strategy
    expected_hosts="""+repr(HOSTS)+"""
    require(spec['host_release']==expected_hosts[system],'per-system historical controller source differs')
    require(spec.get('baseline_source_strategy')==(system if a.strategy else 'correctness-only'),'wrong strategy source spec')"""
assert s.count(marker)==1;(newroot/'bind.py').write_text(s.replace(marker,replace))
s=(HERE/'baseline-until-complete-v1/validate.py').read_text();marker="    require(binding.get('model') in ('7b','14b'),'this gate is the A/C eight-TP1 resident scope')"
s=s.replace(marker,marker+"\n    require(binding['host_release']=="+repr(str(base))+",'C fresh native correctness uses the original shared backend source')")
(newroot/'validate.py').write_text(s)
refs=copy.deepcopy(read(HERE/'baseline-until-complete-v1/manifest.json')['references'])
refs.update({str(p):sha(p) for p in [HERE/'baseline-until-complete-v1/manifest.json',Path(__file__).resolve(),base/'manifest.json',coop/'manifest.json']})
write(newroot/'manifest.json',dict(schema=2,created_s=time.time(),files={str(newroot/n):sha(newroot/n) for n in ['bind.py','validate.py']},references=refs,
 baseline_controller_hosts=HOSTS,only_host_source_difference=changed,ordinary_pd_temporal_backend_byte_identical=True,
 fresh_native_gate_required=True,gate_work_s=390,gate_cleanup_s=90,deadline_s=None,campaign_lifecycle='until_declared_complete_v1'))
# New restore attempt, same frozen Docker/native/power primitives.
s=(HERE/'restore_boundary_baselines_p4.py').read_text().replace('baseline-boundary-restore-p4','baseline-boundary-restore-p4v2').replace('boundary-p4/declaration.json','boundary-p4v2/declaration.json').replace('baseline-until-complete-v1/manifest.json','baseline-until-complete-v2/manifest.json')
s=s.replace("deadline_s=None,campaign_lifecycle='until_declared_complete_v1',executor_release=str(COMMON.parent),", "deadline_s=None,campaign_lifecycle='until_declared_complete_v1',executor_release=str(COMMON.parent),\n        baseline_source_strategy='correctness-only',baseline_controller_hosts="+repr(HOSTS)+",")
(HERE/'restore_boundary_baselines_p4v2.py').write_text(s)
# New measurement runner verifies the per-row declared source and new recovery/gate evidence.
s=(HERE/'run_boundary_baseline_p4.py').read_text().replace('boundary-p4/declaration.json','boundary-p4v2/declaration.json').replace('boundary-baseline-package-p4.json','boundary-baseline-package-p4v2.json').replace('baseline-boundary-restore-p4/deployment-receipt.json','baseline-boundary-restore-p4v2/deployment-receipt.json').replace('boundary-baseline-gate-p4','boundary-baseline-gate-p4v2')
old="    require(binding['host_release']==str(CAMPAIGN.parent/'releases/five-system100-C7B-baseline-v1-runtime'),\n            'baseline runtime changed')"
new="    require(binding['host_release']==d['baseline_controller_hosts'][args.system],'historical per-system baseline runtime changed')"
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("c['cell_id']=c['cell_id'].replace('parallel-rate-p4-explore-','parallel-rate-p4-boundary-')", "require(c['baseline_controller_host']==binding['host_release'],'row controller source differs from binding')")
(HERE/'run_boundary_baseline_p4v2.py').write_text(s)
# New queue preserves the current system order and requires new qualified binding for each source.
s=(HERE/'boundary_baseline_queue_p4.py').read_text().replace('boundary-baseline-queue-p4','boundary-baseline-queue-p4v2').replace('baseline-boundary-restore-p4','baseline-boundary-restore-p4v2').replace('boundary-baseline-package-p4.json','boundary-baseline-package-p4v2.json').replace('boundary-baseline-bootstrap-p4','boundary-baseline-bootstrap-p4v2').replace('baseline-until-complete-v1/','baseline-until-complete-v2/').replace('boundary-baseline-gate-p4','boundary-baseline-gate-p4v2').replace('boundary-p4/declaration.json','boundary-p4v2/declaration.json').replace('boundary-baselines-p4','boundary-baselines-p4v2').replace('boundary-baseline-bindings-p4','boundary-baseline-bindings-p4v2').replace('run_boundary_baseline_p4.py','run_boundary_baseline_p4v2.py')
s=s.replace("                self.command(name+'-bind',[sys.executable,'-B',str(binder),'--spec',str(SPEC)", "                strategy_spec=HERE/'baseline-strategy-specs-p4v2'/(system+'.json')\n                require(read(strategy_spec)['host_release']==declaration['baseline_controller_hosts'][system],'strategy source not predeclared')\n                self.command(name+'-bind',[sys.executable,'-B',str(binder),'--spec',str(strategy_spec)")
(HERE/'boundary_baseline_queue_p4v2.py').write_text(s)
for p in [newroot/'bind.py',newroot/'validate.py',HERE/'restore_boundary_baselines_p4v2.py',HERE/'run_boundary_baseline_p4v2.py',HERE/'boundary_baseline_queue_p4v2.py']:compile(p.read_text(),str(p),'exec')
print(json.dumps(dict(prepared=True,cpu_only=True,baseline_hosts=HOSTS,declared_cells=len(d['cells']),changed_baseline_source_files=changed)))
