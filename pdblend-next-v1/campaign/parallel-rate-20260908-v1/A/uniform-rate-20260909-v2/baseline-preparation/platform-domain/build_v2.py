"""Explicit per-controller service ceiling; all default behavior remains 2520 MHz."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
REPO=ROOT.parents[1]
CLIENT='benchmarks/scripts/bench_vllm.py'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def numeric_sites(source,replacement):
    """Replace audited numeric ceiling sites, never arbitrary source strings."""
    tree=ast.parse(source);parents={c:n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    offsets=[];lines=source.splitlines(keepends=True);starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    for node in ast.walk(tree):
        if not (isinstance(node,ast.Constant) and type(node.value) is int and node.value==2520):continue
        ancestor=node;scope=[]
        while ancestor in parents:
            ancestor=parents[ancestor]
            if isinstance(ancestor,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):scope.append(ancestor.name)
        expression=replacement(tuple(reversed(scope)))
        if expression is None:continue
        start,end=starts[node.lineno-1]+node.col_offset,starts[node.end_lineno-1]+node.end_col_offset
        assert source[start:end]=='2520'
        offsets.append((start,end,expression,node.lineno,tuple(reversed(scope))))
    result=source
    for start,end,expression,_,_ in sorted(offsets,reverse=True):result=result[:start]+expression+result[end:]
    return result,[dict(line=line,scope=scope,old=2520,new=expression) for _,_,expression,line,scope in offsets]


def edits(name,source):
    changes=[]
    def once(old,new):
        nonlocal source
        assert source.count(old)==1,(name,old,source.count(old))
        source=source.replace(old,new);changes.append(dict(old=old,new=new))
    site_mapper={
        'backend.py':lambda scope:'self.max_frequency' if scope and scope[0] in ('ClockOwner','HttpEngineBackend') else None,
        'runtime.py':lambda scope:'self.max_frequency' if scope and scope[0]=='Controller' else None,
        'ecoserve.py':lambda scope:'self.max_frequency' if scope and scope[0]=='EcoServeScheduler' else None,
        'distserve.py':lambda scope:'self.estimator.max_frequency' if scope and scope[0]=='DistServeScheduler' else None,
        'dynamo_topology.py':lambda scope:'self.max_frequency' if scope and scope[0]=='DynamoTopologyPlanner' else None,
        'dynamo.py':lambda scope:'self.estimator.max_frequency' if scope and scope[0]=='DynamoScheduler' else None,
        'frequency.py':lambda scope:'estimator.max_frequency' if scope==('FrequencyPlanner','plan') else None,
        'topology.py':lambda scope:'self.backend.max_frequency' if scope and scope[0]=='TopologyManager' else None,
        'baselines.py':lambda scope:'self.max_frequency' if scope and scope[0]=='DistServeSearch' else None,
    }
    source,sites=numeric_sites(source,site_mapper[name])
    if name=='backend.py':
        once('settle_timeout_s=.3):','settle_timeout_s=.3,max_frequency=2520):')
        once('self.hardware, self.gpus = hardware, tuple(sorted(gpus))','self.max_frequency = max_frequency\n        self.hardware, self.gpus = hardware, tuple(sorted(gpus))')
        once('def __init__(self, instances, session, clocks=None,park_grace_s=.5):','def __init__(self, instances, session, clocks=None,park_grace_s=.5,max_frequency=2520):')
        once('self.instances = {i["id"]:i for i in instances}', 'self.max_frequency = clocks.max_frequency if clocks is not None else max_frequency\n        self.instances = {i["id"]:i for i in instances}')
    elif name=='runtime.py':
        once('self.config=config','self.config=config\n        self.max_frequency=config.get("max_service_frequency_mhz",2520)\n        if type(self.max_frequency) is not int or self.max_frequency<=0:\n            raise ValueError("explicit positive integer service frequency required")')
        once('self.planner=JointPlanner(self.profiles,','self.planner=JointPlanner(self.profiles,max_frequency=self.max_frequency,')
        once("lower=config.get('eco_macro_lower',2),upper=config.get('eco_macro_upper',3))", "lower=config.get('eco_macro_lower',2),upper=config.get('eco_macro_upper',3),max_frequency=self.max_frequency)")
        once("clock_settle_s=config.get('clock_settle_s',.3),topology=self.interconnect)","clock_settle_s=config.get('clock_settle_s',.3),topology=self.interconnect,max_frequency=self.max_frequency)")
        once("self.dynamo_scheduler=(DynamoScheduler(self.profiles,config['dynamo_assignments'],", "self.dynamo_scheduler=(DynamoScheduler(self.profiles,config['dynamo_assignments'],max_frequency=self.max_frequency,")
        once('clocks=await asyncio.to_thread(ClockOwner,hardware,gpus)','clocks=await asyncio.to_thread(ClockOwner,hardware,gpus,max_frequency=self.max_frequency)')
        once("park_grace_s=self.config.get('park_grace_s',.5))", "park_grace_s=self.config.get('park_grace_s',.5),max_frequency=self.max_frequency)")
        once("[TopologyCost(**c) for c in config['topology_costs']],","[TopologyCost(**c) for c in config['topology_costs']],max_frequency=self.max_frequency,")
    elif name=='ecoserve.py':
        once('def __init__(self,profiles,instances,*,lower=2,upper=3):','def __init__(self,profiles,instances,*,lower=2,upper=3,max_frequency=2520):')
        once('self.profiles=profiles','self.max_frequency=max_frequency\n        self.profiles=profiles')
    elif name=='distserve.py':
        once('clock_settle_s=.3,topology=None):','clock_settle_s=.3,topology=None,max_frequency=2520):')
        once('JointPlanner(profiles,transfers,dvfs=False,','JointPlanner(profiles,transfers,max_frequency=max_frequency,dvfs=False,')
    elif name=='dynamo.py':
        once('output_cuts=(99,349),clock_settle_s=.3,frequency_costs=()):','output_cuts=(99,349),clock_settle_s=.3,frequency_costs=(),max_frequency=2520):')
        once('JointPlanner(profiles,allow_pd=False,dvfs=True,','JointPlanner(profiles,max_frequency=max_frequency,allow_pd=False,dvfs=True,')
    elif name=='dynamo_topology.py':
        once('def __init__(self,profiles,costs,*,error_fraction=.3,gpu_count=8,park_grace_s=.5):','def __init__(self,profiles,costs,*,error_fraction=.3,gpu_count=8,park_grace_s=.5,max_frequency=2520):')
        once('self.profiles=profiles;self.costs=tuple(costs)','self.max_frequency=max_frequency\n        self.profiles=profiles;self.costs=tuple(costs)')
    elif name=='baselines.py':
        once('def __init__(self,profiles,transfers,gpu_count=8,topology=None):','def __init__(self,profiles,transfers,gpu_count=8,topology=None,max_frequency=2520):')
        once('self.profiles,self.transfers,self.gpu_count=profiles,tuple(transfers),gpu_count', 'self.max_frequency=max_frequency\n        self.profiles,self.transfers,self.gpu_count=profiles,tuple(transfers),gpu_count')
        once("online_profile_points(self.profiles,'prefill',input_tokens,input_tokens+1)", "online_profile_points(self.profiles,'prefill',input_tokens,input_tokens+1,self.max_frequency)")
        once("online_profile_points(self.profiles,'decode',input_tokens,input_tokens+predicted_output)", "online_profile_points(self.profiles,'decode',input_tokens,input_tokens+predicted_output,self.max_frequency)")
    # Reversing only the declared parameter plumbing and ceiling substitutions
    # must reconstruct the original bytes exactly, including every branch.
    return source,dict(numeric_ceiling_sites=sites,explicit_parameter_plumbing=changes)


def build(parent,out):
    assert not out.exists()
    manifest=read(parent/'manifest.json')
    changed={};newfiles={}
    names={'backend.py','runtime.py','ecoserve.py','distserve.py','dynamo.py','frequency.py','topology.py','baselines.py','dynamo_topology.py'}
    for relative,digest in manifest['files'].items():
        old=parent/relative;assert sha(old)==digest
        data=old.read_bytes()
        if relative.startswith('src/ecopadg/serving/') and Path(relative).name in names:
            transformed,proof=edits(Path(relative).name,data.decode())
            ast.parse(transformed)
            proof['parent_sha256']=digest
            changed[relative]=proof
            data=transformed.encode()
        elif relative==CLIENT:
            data=(ROOT/'common/token-evidence-v2/bench_vllm.py').read_bytes()
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        newfiles[relative]=sha(dest)
    save(out/'manifest.json',dict(schema='newA-legacy-explicit-service-frequency-v1',parent_release=str(parent),
        parent_manifest_sha256=sha(parent/'manifest.json'),files=newfiles,
        max_service_frequency_mhz_default=2520,explicit_target_domain_mhz=[900,1500,2100],
        configuration_key='max_service_frequency_mhz',platform_adaptation=changed,
        collector=ref(ROOT/'common/token-evidence-v2/bench_vllm.py'),builder=ref(Path(__file__)),
        qualified=False,formal_eligible=False))
    patch=''
    for rel in changed:
        patch+=''.join(difflib.unified_diff((parent/rel).read_text().splitlines(True),(out/rel).read_text().splitlines(True),fromfile=str(parent/rel),tofile=str(out/rel)))
    (out/'platform.diff').write_text(patch)
    return ref(out/'manifest.json')


if __name__=='__main__':
    refs={}
    for variant in ('baseline','baseline-eco-drain'):
        parent=REPO/'releases'/('five-system100-A14B-'+variant+'-v1-runtime')
        refs[variant]=build(parent,HERE/('runtime-'+variant+'-002'))
    save(HERE/'candidate-hosts-002.json',refs)
    print(json.dumps(refs,indent=2))
