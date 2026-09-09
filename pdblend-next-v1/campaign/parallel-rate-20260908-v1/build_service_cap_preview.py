"""Build a CPU-only candidate for a measured lower service clock ceiling.

No production manifests/configs are modified and no GPU is touched here.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'hosts/14b-fixed-p4'
OUT=ROOT/'drafts/service-cap-v1'

def once(s,a,b):
    if s.count(a)!=1:raise ValueError((a,s.count(a)))
    return s.replace(a,b)

def build():
    if OUT.exists():raise FileExistsError(OUT)
    manifest=json.loads((PARENT/'manifest.json').read_text())
    files={}
    for n,h in manifest['files'].items():
        raw=(PARENT/n).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==h
        files[n]=raw.decode() if n.endswith('.py') else raw
    def get(n):return files['src/ecopadg/serving/'+n+'.py']
    def put(n,s):ast.parse(s);files['src/ecopadg/serving/'+n+'.py']=s

    s=get('runtime')
    s=once(s,'        self.state=StateManager(','''        self.max_service_frequency_mhz=config.get('max_service_frequency_mhz',2520)
        if (type(self.max_service_frequency_mhz) is not int
                or not 0<self.max_service_frequency_mhz<=2520):
            raise ValueError('service frequency ceiling must be a positive integer at most 2520 MHz')
        if self.max_service_frequency_mhz!=2520 and (
                not self.strategy.startswith('pdblend')
                or config.get('measured_frequency_write_guard_v1') is not True
                or config.get('allow_pd') is not False or config.get('topology')
                or config.get('slow_topology') or config.get('dynamic_pools')
                or any(i.get('role')!='mixed' for i in config['instances'])):
            raise ValueError('lower service ceiling currently requires guarded fixed mixed PDB')
        self.state=StateManager(''')
    s=once(s,'        self.interconnect=', '''        if self.max_service_frequency_mhz!=2520:
            # A ceiling is a measured operating point, not a tolerance change.
            costs={(c['tp'],c['source_mhz'],c['target_mhz'])
                   for c in config.get('frequency_costs',()) if c.get('source_sha256')}
            for i in config['instances']:
                frequencies={f for f in self.profiles.frequencies(i['role'],i['tp'])
                             if f<=self.max_service_frequency_mhz}
                if self.max_service_frequency_mhz not in frequencies:
                    raise ValueError('service ceiling lacks a measured profile frequency')
                if any((i['tp'],a,b) not in costs for a in frequencies for b in frequencies if a!=b):
                    raise ValueError('service ceiling lacks measured reachable transition costs')
        self.interconnect=''')
    s=once(s,'        self.planner=JointPlanner(self.profiles,','        self.planner=JointPlanner(self.profiles,max_frequency=self.max_service_frequency_mhz,')
    s=once(s,'clocks=await asyncio.to_thread(ClockOwner,hardware,gpus)',
        'clocks=await asyncio.to_thread(ClockOwner,hardware,gpus,max_frequency=self.max_service_frequency_mhz)')
    s=s.replace('clocks.set(sorted(gpus),2520)','clocks.set(sorted(gpus),self.max_service_frequency_mhz)')
    s=once(s,"park_grace_s=self.config.get('park_grace_s',.5))", "park_grace_s=self.config.get('park_grace_s',.5),max_frequency=self.max_service_frequency_mhz)")
    s=s.replace('FrequencyAction(i.instance_id,2520)','FrequencyAction(i.instance_id,self.max_service_frequency_mhz)')
    put('runtime',s)

    s=get('backend')
    s=once(s,'settle_timeout_s=.3):','settle_timeout_s=.3, *, max_frequency=2520):')
    s=once(s,'        self.hardware, self.gpus = hardware, tuple(sorted(gpus))', '''        if type(max_frequency) is not int or not 0<max_frequency<=2520:
            raise ValueError('invalid clock-owner service ceiling')
        self.max_frequency=max_frequency
        self.hardware, self.gpus = hardware, tuple(sorted(gpus))''')
    s=once(s,'        # Reserve clock intent before awaiting the writer.', '''        if type(frequency) is not int or not 0<frequency<=self.max_frequency:
            raise ValueError('clock target exceeds configured measured service ceiling')
        # Reserve clock intent before awaiting the writer.''')
    s=once(s,'def __init__(self, instances, session, clocks=None,park_grace_s=.5):',
        'def __init__(self, instances, session, clocks=None,park_grace_s=.5, *, max_frequency=2520):')
    s=once(s,'        self.instances = {i["id"]:i for i in instances}', '''        if type(max_frequency) is not int or not 0<max_frequency<=2520:
            raise ValueError('invalid backend service ceiling')
        if clocks is not None and getattr(clocks,'max_frequency',2520)!=max_frequency:
            raise ValueError('backend and clock-owner service ceilings disagree')
        self.max_frequency=max_frequency
        self.instances = {i["id"]:i for i in instances}''')
    # Keep literal defaults; replace every operational hardcoded high clock.
    for a,b in [('gpus,2520,','gpus,self.max_frequency,'),('item[\'gpus\'],2520,',"item['gpus'],self.max_frequency,"),
                ('!=2520','!=self.max_frequency'),('gpu,2520,','gpu,self.max_frequency,'),
                ('member,2520,','member,self.max_frequency,'),('=2520\n','=self.max_frequency\n'),
                ('{i:2520 for','{i:self.max_frequency for'),('self.frequency.get(i,2520)','self.frequency.get(i,self.max_frequency)'),
                ('                2520 if started','                self.max_frequency if started'),
                ('self.clocks.applied.get(g,2520)','self.clocks.applied.get(g,self.max_frequency)'),
                ('commanded=2520,','commanded=self.max_frequency,')]:
        s=s.replace(a,b)
    put('backend',s)

    s=get('planner')
    s=once(s,'        self.max_frequency = max_frequency', '''        if type(max_frequency) is not int or not 0<max_frequency<=2520:
            raise ValueError('invalid planner service ceiling')
        self.max_frequency = max_frequency''')
    s=once(s,'        context = max(resident_context(instance)', '        if freq>self.max_frequency:return None\n        context = max(resident_context(instance)')
    s=once(s,'self.profiles.frequencies(d.role,d.tp) if self.dvfs',
        'tuple(f for f in self.profiles.frequencies(d.role,d.tp) if f<=self.max_frequency) if self.dvfs')
    put('planner',s)

    s=get('frequency')
    s=once(s,'    for frequency in estimator.profiles.frequencies(instance.role,instance.tp):',
        '    for frequency in estimator.profiles.frequencies(instance.role,instance.tp):\n        if frequency>getattr(estimator,\'max_frequency\',2520):continue')
    s=once(s,'            for frequency in estimator.profiles.frequencies(i.role,i.tp):',
        '            for frequency in estimator.profiles.frequencies(i.role,i.tp):\n                if frequency>getattr(estimator,\'max_frequency\',2520):continue')
    s=once(s,'        expires=now+estimator.telemetry_ttl_s','        maximum=getattr(estimator,\'max_frequency\',2520)\n        expires=now+estimator.telemetry_ttl_s')
    s=s.replace('i.frequency_mhz!=2520','i.frequency_mhz!=maximum').replace('if choices else 2520','if choices else maximum').replace('chosen==2520','chosen==maximum')
    s=s.replace('recovery_actions(estimator,recoveries,now,expires)','recovery_actions(estimator,recoveries,now,expires,maximum=maximum)')
    s=once(s,"frequencies={p['frequency_mhz'] for p in profiles['points'] if p['tp']==tp}",
        "frequencies={p['frequency_mhz'] for p in profiles['points'] if p['tp']==tp and p['frequency_mhz']<=config.get('max_service_frequency_mhz',2520)}")
    put('frequency',s)

    s=get('pending_frequency')
    s=once(s,'        for target in estimator.profiles.frequencies(instance.role,instance.tp):',
        '        for target in estimator.profiles.frequencies(instance.role,instance.tp):\n            if target>getattr(estimator,\'max_frequency\',2520):continue')
    put('pending_frequency',s)
    s=get('idle_admission')
    s=once(s,'        frequencies=controller.planner.profiles.frequencies(instance.role,instance.tp)',
        '        frequencies=tuple(f for f in controller.planner.profiles.frequencies(instance.role,instance.tp)\n                          if f<=controller.planner.max_frequency)')
    put('idle_admission',s)
    s=get('physical_frequency')
    s=once(s,'    if bootstrap is not None:', '''    if frequency is not None and frequency>getattr(controller.planner,'max_frequency',2520):
        return dict(result,error='target exceeds configured measured service ceiling')
    if bootstrap is not None:''')
    put('physical_frequency',s)

    for n,raw in files.items():
        p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(raw.encode() if isinstance(raw,str) else raw)
    (OUT/'PREVIEW.json').write_text(json.dumps(dict(parent=str(PARENT),gpu_qualified=False,
        production_use_permitted=False,changed_files=[n for n,r in files.items()
            if (OUT/n).read_bytes()!=(PARENT/n).read_bytes()]),indent=2)+'\n')
    print(OUT)

if __name__=='__main__':build()
