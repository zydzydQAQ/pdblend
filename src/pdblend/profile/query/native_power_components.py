"""Refit and replay a new native power component; a pilot cannot self-promote."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import math
import statistics

from pdblend.profile.collection.native_power_audit import audit_power_window
from pdblend.profile.collection.native_timing_audit import finite, need
from pdblend.profile.collection.native_timing_plan import digest

KIND='pdblend_native_power_table_v1'
EVIDENCE='pdblend-native-power-calibration-evidence/v1'
LIMITS=dict(mean_relative_error=.10,max_relative_error=.15,each_window_relative_error=.10)


def fit_nodes(rows,identity):
    groups=defaultdict(list)
    for raw,audit,ref in rows:
        p=raw['point'];groups[(p['role'],p['frequency_mhz'],p['batch'],p['prompt_tokens'])].append((raw,audit,ref))
    nodes=[]
    for (role,f,b,n),values in sorted(groups.items()):
        need(len(values)==3 and {r['point']['repeat'] for r,_,_ in values}=={0,1,2},
             'power training needs three separate repetitions per point')
        context=[a['effective_context_tokens'] if role=='decode' else n for _,a,_ in values]
        nodes.append(dict(role=role,frequency_mhz=f,batch=b,prompt_tokens=n,
            context_min=min(context),context_max=max(context),power_w=statistics.mean(a['power_w'] for _,a,_ in values),
            raw_sha256=sorted(ref['sha256'] for _,_,ref in values)))
    return dict(kind=KIND,identity=deepcopy(identity),nodes=nodes,
        batch_interpolation='linear_ge4_exact_1_2_3',frequency_interpolation=False,
        power_scope='prefill_request_cycle_and_continuous_pure_decode_at_actual_mean_context',
        active_prefill_kernel_power_qualified=False)


class NativePowerTable:
    """Measured domains only; low fractional occupancy needs another revision."""
    def __init__(self,candidate):
        need(candidate.get('kind')==KIND and candidate.get('batch_interpolation')=='linear_ge4_exact_1_2_3'
             and candidate.get('frequency_interpolation') is False,'unknown native power candidate semantics')
        self.candidate=deepcopy(candidate);self.groups=defaultdict(list)
        for node in candidate['nodes']:
            need(node['role'] in ('prefill','decode') and node['frequency_mhz'] in (1500,2520)
                 and type(node['batch']) is int and 1<=node['batch']<=32
                 and all(finite(node[k]) and node[k]>0 for k in ('context_min','context_max','power_w'))
                 and node['context_min']<=node['context_max'],'invalid actual power node')
            self.groups[(node['role'],node['frequency_mhz'],node['batch'])].append(node)
        for nodes in self.groups.values():
            nodes.sort(key=lambda n:n['context_min'])
            need(all(a['context_max']<b['context_min'] for a,b in zip(nodes,nodes[1:])),
                 'overlapping actual mean-context power bands')

    @staticmethod
    def _curve(nodes,context):
        if not nodes or not finite(context) or not nodes[0]['context_min']<=context<=nodes[-1]['context_max']:
            raise ValueError('missing_profile: native power context outside actual measured domain')
        for row in nodes:
            if row['context_min']<=context<=row['context_max']:return row['power_w']
        for a,b in zip(nodes,nodes[1:]):
            if a['context_max']<context<b['context_min']:
                weight=(context-a['context_max'])/(b['context_min']-a['context_max'])
                return a['power_w']+weight*(b['power_w']-a['power_w'])
        raise ValueError('missing_profile: native power interpolation gap')

    def predict(self,role,batch,context,frequency):
        need(finite(batch) and finite(context) and finite(frequency),'missing_profile: nonfinite power query')
        if role=='prefill':
            need(batch==1,'missing_profile: native prefill power supports batch1 only')
        nodes=self.groups.get((role,frequency,batch))
        if nodes:return self._curve(nodes,context)
        if role!='decode' or batch<4:
            raise ValueError('missing_profile: low fractional occupancy has no qualified power model')
        batches=sorted(b for r,f,b in self.groups if r==role and f==frequency and b>=4)
        lower=[b for b in batches if b<batch];upper=[b for b in batches if b>batch]
        if not lower or not upper:raise ValueError('missing_profile: native power batch outside measured domain')
        lo,hi=max(lower),min(upper)
        a=self._curve(self.groups[(role,frequency,lo)],context)
        b=self._curve(self.groups[(role,frequency,hi)],context)
        return a+(batch-lo)/(hi-lo)*(b-a)


def replay_power(evidence,resolver,identity):
    """No saved qualification flag is trusted: samples refit the frozen candidate."""
    need(evidence.get('schema')==EVIDENCE,
         'pure power pilot has no training candidate or independent holdout; native calibration evidence required')
    need(evidence.get('identity')==identity,'power component model/source identity differs')
    plan=resolver.read(evidence['plan']);candidate=resolver.read(evidence['candidate'])
    need(plan.get('schema')=='pdblend-native-power-calibration-plan/v1'
         and plan.get('evaluation_used_for_selection') is False and plan.get('identity')==identity,
         'bound non-evaluation power calibration design required')
    expected={digest(p):p for p in plan['points']}
    need(expected and len(expected)==len(plan['points']),'duplicate or missing power calibration points')
    training=[];holdout=[];seen=set();intervals=[]
    for ref in evidence['windows']:
        raw=resolver.read(ref);point=dict(raw['point']);repeat=point.pop('repeat')
        frozen_candidate=point.pop('candidate_sha256',None)
        key=digest(point);need(key in expected,'raw power point absent from frozen calibration design')
        need(type(repeat)is int and 0<=repeat<3 and (key,repeat) not in seen,'duplicate/missing power repetition')
        seen.add((key,repeat));purpose=point.get('purpose')
        need(purpose in ('training','holdout') and point['seed']==(9701 if purpose=='training' else 9702),
             'pilot or shared-seed power windows cannot replace independent calibration/holdout')
        need(all(raw['capability'].get(k)==identity[k] for k in identity if k!='system'),
             'actual native power capability differs from component identity')
        audit=audit_power_window(raw)
        if purpose=='holdout':
            need(frozen_candidate==evidence['candidate']['sha256'],'held-out power window does not bind frozen candidate')
            holdout.append((raw,audit,ref))
        else:
            need(frozen_candidate is None,'training cannot use the frozen holdout candidate')
            training.append((raw,audit,ref))
        intervals.append((raw['settle_started_s'],raw['drain']['response_at_s']))
    need(seen=={(key,r) for key in expected for r in range(3)},'power training/holdout raw inventory incomplete')
    need(training and holdout,'pure power training and independent holdout both required')
    ordered=sorted(intervals);need(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),
        'serial native power calibration windows overlap')
    frozen=evidence.get('candidate_frozen_s')
    need(finite(frozen) and max(r['drain']['response_at_s'] for r,_,_ in training)<=frozen
         <min(r['settle_started_s'] for r,_,_ in holdout),'candidate freeze must precede every held-out workload')
    rebuilt=fit_nodes(training,identity)
    need(candidate==rebuilt,'power candidate differs from raw training-only refit')
    table=NativePowerTable(rebuilt);errors=[];coverage=[]
    for raw,audit,ref in holdout:
        p=raw['point'];context=audit.get('effective_context_tokens',p['prompt_tokens'])
        predicted=table.predict(p['role'],p['batch'],context,p['frequency_mhz'])
        error=abs(predicted/audit['power_w']-1);errors.append(error)
        coverage.append(dict(role=p['role'],frequency_mhz=p['frequency_mhz'],batch=p['batch'],context=context,
            relative_error=error,raw_sha256=ref['sha256']))
    need({(r['role'],r['frequency_mhz']) for r in coverage}=={(r,f) for r in ('prefill','decode') for f in (1500,2520)},
         'power holdout must cover both roles at both exact frequencies')
    need(statistics.mean(errors)<=LIMITS['mean_relative_error'] and max(errors)<=LIMITS['max_relative_error']
         and all(e<=LIMITS['each_window_relative_error'] for e in errors),'native pure-power independent holdout failed')
    return dict(candidate=rebuilt,component_qualified=True,raw_windows=len(seen),holdout=coverage,
        holdout_limits=LIMITS,low_fractional_batch_qualified=False,
        serving_energy_composition_qualified=False),table
