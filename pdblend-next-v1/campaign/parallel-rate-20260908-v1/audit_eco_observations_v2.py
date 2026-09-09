"""Retain C/B audits and independently replay A's actual fresh 55-request gate."""
import csv
import json
from pathlib import Path
import audit_eco_observations_v1 as previous
import a_eco_scope_v1 as a_scope

_cache={}


def a_execution(p,cp,binding):
    matches=[]
    for path,digest in binding['files'].items():
        if Path(path).name!='release.json':continue
        doc=p.read(path)
        if doc.get('schema')=='A-Eco-scoped-qualified-execution-release-v1' and doc.get('declaration')==cp['declaration']:
            p.need(p.sha(path)==digest,'A actual Eco execution release changed')
            matches.append((dict(path=path,sha256=digest),doc))
    p.need(len(matches)==1,'A Eco observation must freeze one actual scoped execution release')
    ref,release=matches[0]
    p.need(release['model']=='14b' and release['ready_for_gpu'] is True
           and release['execution_rules']==cp['execution_rules']
           and release['host_release']==binding['host_release']
           and release['source_package']==dict(path=str(a_scope.PACKAGE),sha256=a_scope.PACKAGE_SHA),
           'A Eco observation changed reviewed source, scope or execution rules')
    for path,digest in release['files'].items():
        p.need(p.sha(path)==digest and binding['files'].get(path)==digest,
               'A Eco execution release source or predecessor detached from observation')
    frozen=p.checked(release['binding'])
    expected=dict(frozen,output=binding['output'],files=dict(frozen['files']))
    expected['files'].update({ref['path']:ref['sha256'],cp['declaration']['path']:cp['declaration']['sha256']})
    p.need(binding==expected,'A observation differs from the exact scoped release binding')
    rules=p.checked(cp['execution_rules'])
    p.need(rules['schema']=='A-Eco-fixed100-arrival-engineering-v1'
           and rules['scope']==cp['declaration']
           and rules['arrival_window_s']==100 and rules['request_timeout_s']==120
           and rules['cleanup_local_budget_s']==90 and rules['all8gpu_power'] is True
           and rules['max_dispatch_lateness_s']==1 and rules['p99_dispatch_lateness_s']==.1
           and rules['max_handler_lateness_s']==1,'A Eco original measurement and arrival rules changed')
    module,_=a_scope.contract(p);qref=release['qualification_binding'];base=p.checked(qref)
    for key in ('instances','host_release','correctness_evidence','mechanism_proof','deployment_receipt'):
        p.need(binding[key]==base[key],'A observation changed its actual fresh qualification: '+key)
    for path,digest in base['files'].items():
        p.need(p.sha(path)==digest and binding['files'].get(path)==digest,
               'A fresh qualification or measurement input no longer pinned')
    for dataset,path in base['configs'].items():
        actual_path=binding['configs'][dataset]
        p.need(binding['files'][actual_path]==p.sha(actual_path),'A actual Eco config not frozen')
        original,actual=p.read(path),p.read(actual_path)
        p.need({k:v for k,v in original.items() if k!='journal'}=={k:v for k,v in actual.items() if k!='journal'},
               'A Eco observation changed qualified serving configuration')
    if qref['sha256'] not in _cache:
        _,proof=module.qualification(qref)
        p.need(proof['request_count']==55 and proof['legacy_gate_passed'] is True
               and proof['raw_gate_recomputed'] is True,'A native qualification not actually complete')
        _cache[qref['sha256']]=proof
    return dict(execution_release=ref,qualified_binding=qref,
                independently_recomputed=True,qualification=_cache[qref['sha256']])


def verify(p,cp,binding,summary,directory,checkpoint):
    if cp['row']['model']!='14b':return previous.verify(p,cp,binding,summary,directory,checkpoint)
    p.need(summary['runtime_error'] is None,'A Eco controller failure needs an independent diagnosis')
    bench=list(csv.DictReader((directory/'bench.csv').open()))
    events=[json.loads(s) for s in (directory/'control.jsonl').read_text().splitlines()]
    timing=previous.timing(p,bench,events)
    qualification=a_execution(p,cp,binding)
    p.need(summary['work_complete'] and summary['failed_requests']==summary['request_timeouts']==0
           and all(r['success']=='1' for r in bench),
           'A Eco incomplete work awaits an independent failure diagnosis')
    return dict(classification='valid_complete_work_observation',classified_native_queue_refusals=0,
                timing=timing,native_qualification=qualification,original_raw_unchanged=True)
