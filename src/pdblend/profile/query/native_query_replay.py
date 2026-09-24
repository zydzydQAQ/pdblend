"""Re-enumerate every tuning group; unsupported alternatives are not samples.

A caller cannot qualify a hand-picked list of successful scalar queries. The
original bound tuning confirmations supply requests and independently confirmed
rates; this module re-runs the unchanged canonical policy for all twelve groups.
"""
from __future__ import annotations
from dataclasses import asdict,replace
import math

from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_timing_plan import digest

DATASETS=('alpaca','sharegpt','longbench')
SCALES=(.25,.5,.75,1.)
QUERIES={'prefill_seconds','prefill_marginal_seconds','prefill_power_w','prefill_energy_j','step_seconds',
    'decode_supported','decode_power_supported','decode_power_w','token_energy_j','mixed_power_supported',
    'mixed_power_w','static_power_w','wake_seconds','transfer_seconds'}


def _json(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_float':repr(value)}
    if isinstance(value,(tuple,list)):return [_json(v) for v in value]
    if isinstance(value,dict):return {k:_json(v) for k,v in value.items()}
    return value


class QueryLog:
    def __init__(self,model):self.base=model;self.rows=[]
    def __getattr__(self,name):
        method=getattr(self.base,name)
        if name not in QUERIES or not callable(method):return method
        def invoke(*args,**kwargs):
            row=dict(method=name,args=_json(args),kwargs=_json(kwargs));self.rows.append(row)
            try:value=method(*args,**kwargs)
            except ValueError as exc:
                # PoolPlanner explicitly handles these profile-domain errors.
                # Other exceptions must abort the full replay, never disappear.
                row.update(error=str(exc),unsupported=True)
                if 'missing_profile:' not in str(exc) and 'outside measured coverage' not in str(exc):
                    raise RuntimeError('unexpected profile query error: '+str(exc)) from exc
                raise
            if isinstance(value,bool):row.update(value=value,unsupported=not value)
            else:
                if not finite(value) or value<0:raise RuntimeError('profile returned an invalid numeric prediction')
                row.update(value=value,unsupported=False)
            return value
        return invoke


def tuning_groups(ledger_ref,provenance_ref,resolver,model_id,tp,frequency_domain=None):
    """Reconstruct model-owned requests/rates, retaining the original SHA chain."""
    ledger=resolver.read(ledger_ref);provenance=resolver.read(provenance_ref)
    from pdblend.profile.collection.native_frequency_domain import validate_domain,domain_fields
    frequencies=[1500,2520]
    if frequency_domain is not None:
        domain=validate_domain(frequency_domain);fields=domain_fields(domain)
        need(domain['model_id']==model_id and domain['tp']==tp
             and all(ledger.get(k)==provenance.get(k)==v for k,v in fields.items())
             and ledger.get('frequency_domain_ref')==provenance.get('frequency_domain_ref')
             and resolver.read(ledger['frequency_domain_ref'])==domain,
             'actual tuning query frequency domain binding differs')
        frequencies=domain['frequencies_mhz']
    else:
        need(not any(k in value for value in (ledger,provenance)
                     for k in ('frequency_domain','frequency_domain_sha256','frequency_domain_ref')),
             'new frequency tuning requires an explicit domain consumer')
    need(ledger.get('schema')=='pdblend-offline-query-ledger/v2' and ledger.get('evaluation_read') is False
         and ledger.get('min_m_floor_overridden') is False and provenance.get('evaluation_read') is False
         and any(r.get('sha256')==ledger_ref['sha256'] for r in provenance['outputs'].values()),
         'original actual-query ledger/provenance binding differs')
    size=model_id.split('-')[1].lower();inputs=provenance['inputs'][size]
    anchor=resolver.read(inputs['rate_anchor'])
    need(anchor.get('model_id')==model_id and anchor.get('evaluation_used_for_selection') is False,
         'tuning anchor identity or split differs')
    rows=[r for r in ledger['ledgers'] if r['model_id']==model_id]
    need(len(rows)==12 and {(r['dataset'],r['rate_scale']) for r in rows}==
         {(d,s) for d in DATASETS for s in SCALES},'complete twelve-group tuning inventory required')
    groups=[]
    for dataset in DATASETS:
        owner=anchor
        if dataset=='longbench' and inputs.get('longbench_recovery'):
            owner=resolver.read(inputs['longbench_recovery']['completion'])
            preflight=resolver.read(inputs['longbench_recovery']['preflight'])
            from pdblend.bench.longbench_mixed_combo import classify_recovery
            need(owner['prior_inputs']['receipts']['completion']==inputs['rate_anchor'] and
                 classify_recovery(owner,preflight,resolver.path(inputs['longbench_recovery']['completion']['path']).parent)=='confirmed',
                 'LongBench recovery lacks its independently confirmed anchor')
        selected=owner['anchors'][dataset];refs=inputs['datasets'][dataset]
        confirmation=resolver.read(refs['confirmation']);trace=resolver.read(refs['tuning_trace'])
        need(selected['model_id']==model_id and selected['tp']==tp and selected['pp']==1
             and refs['confirmation']['sha256']==selected['confirmation_sha256']
             and confirmation.get('split')=='tuning' and confirmation.get('dataset')==dataset
             and confirmation['metrics'].get('passed') is True and confirmation['rate_rps']==selected['base_rate_rps']
             and confirmation['trace_sha256']==refs['tuning_trace']['sha256']
             and trace['seed']==confirmation['seed']==owner['tuning_seed'],
             'confirmed tuning trace/model/rate lineage differs')
        need(trace['requests'] and all(r.get('source')==dataset for r in trace['requests']),
             'tuning request cohort does not belong to the named dataset')
        for scale in SCALES:
            row=next(r for r in rows if (r['dataset'],r['rate_scale'])==(dataset,scale))
            target=selected['base_rate_rps']*scale
            need(row['selection_split']=='tuning' and row['tp']==tp and row['pp']==1
                 and row['min_m_instances']==4 and row['engine_max_num_seqs']==32 and row['slots']==8//tp
                 and row['frequency_scope']==frequencies and row['target_rate_rps']==target
                 and (frequency_domain is None or row.get('frequency_domain_sha256')==fields['frequency_domain_sha256']),
                 'actual tuning policy or rate scope differs')
            groups.append(dict(dataset=dataset,rate_scale=scale,target_rate_rps=target,
                               trace=trace,trace_binding=refs['tuning_trace'],confirmation=refs['confirmation']))
    return groups


def enumerate_groups(model,groups):
    from pdblend.bench.client import Request,SLOS
    from pdblend.bench.run import offline_forecast
    from pdblend.online.policies import get_policy
    from pdblend.planner.pool import PoolPlanner,PlannerConfig,SLO
    results=[]
    for group in groups:
        fc=replace(offline_forecast([Request(**r) for r in group['trace']['requests']]),rate_rps=group['target_rate_rps'])
        config=get_policy('pdblend').planner_config(PlannerConfig(slots=8//model.tp,slo=SLO(*SLOS[group['dataset']]),
                                                                 freqs=(1500,2520),max_num_seqs=32))
        need(config.min_m_instances==4,'canonical PD policy floor changed')
        logged=QueryLog(model);planner=PoolPlanner(logged,config);candidates=planner.candidates(fc)
        need(candidates,'no feasible qualified candidate for '+group['dataset']+' scale '+str(group['rate_scale']))
        chosen=candidates[0]
        need(finite(chosen.power_w) and chosen.power_w>0 and finite(chosen.ttft_s) and finite(chosen.tpot_s),
             'chosen candidate lacks finite timing/energy')
        # Re-evaluate the selected deployment on a fresh planner without cached
        # roles or an unsupported fallback. Peak search may reject alternatives.
        chosen_log=QueryLog(model)
        verified=PoolPlanner(chosen_log,config).evaluate(chosen.counts,chosen.f_P,chosen.f_D,chosen.f_M,chosen.tau,fc)
        need(verified is not None and asdict(verified)==asdict(chosen),'chosen candidate does not reproduce')
        results.append(dict(dataset=group['dataset'],rate_scale=group['rate_scale'],target_rate_rps=group['target_rate_rps'],
            trace=group['trace_binding'],confirmation=group['confirmation'],min_m_instances=4,
            feasible_candidates=len(candidates),candidates_sha256=digest([asdict(c) for c in candidates]),
            chosen=asdict(chosen),queries=logged.rows,chosen_queries=chosen_log.rows,
            unsupported_alternatives=sum(r['unsupported'] for r in logged.rows)))
    return results


def build_query_receipt(ledger_ref,provenance_ref,resolver,model,candidate_sha):
    groups=tuning_groups(ledger_ref,provenance_ref,resolver,model.model,model.tp)
    return dict(schema='pdblend-native-composed-query-ledger/v2',profile_candidate_sha256=candidate_sha,
        original_ledger=ledger_ref,original_provenance=provenance_ref,evaluation_read=False,min_m_floor_overridden=False,
        ledgers=enumerate_groups(model,groups))


def replay_queries(ref,resolver,model,candidate_sha):
    evidence=resolver.read(ref)
    need(evidence.get('schema')=='pdblend-native-composed-query-ledger/v2',
         'fresh full-candidate replay required; scalar-only query receipts are insufficient')
    rebuilt=build_query_receipt(evidence['original_ledger'],evidence['original_provenance'],resolver,model,candidate_sha)
    need(rebuilt==evidence,'complete tuning enumeration or selected candidate differs from raw replay')
    return dict(groups=12,datasets=list(DATASETS),rate_scales=list(SCALES),
        unsupported_alternatives=sum(r['unsupported_alternatives'] for r in rebuilt['ledgers']),
        chosen_candidates=[{k:r[k] for k in ('dataset','rate_scale','target_rate_rps','chosen','trace')} for r in rebuilt['ledgers']],
        unsupported_branches_qualified=False,complete_enumeration=True)
