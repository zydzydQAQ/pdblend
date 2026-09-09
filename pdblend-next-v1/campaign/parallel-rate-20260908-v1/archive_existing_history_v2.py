"""Archive existing history only; the user excluded new SLO-scale measurements."""
import copy
import json
from pathlib import Path
import time
import append_historical_suffix_v1 as shared
import historical_comparison_overlay_v2 as eligibility

ROOT=Path(__file__).resolve().parent
SCOPE_SHA='fcc178ccd6360a0e749e589ed8f396362b0f7e087344e21c3763d514ea963ae7'


def main():
    current=shared.current;p=current.load_audit().p
    base=ROOT/'reports/historical-scale-before-suffix-001'
    out=ROOT/'reports/historical-existing-fixed-slo-scope-002'
    p.need(not out.exists(),'immutable historical archive output exists')
    paths=[Path(__file__),Path(shared.__file__),Path(eligibility.__file__),shared.HISTORICAL_CODE]
    sources={str(x):current.sha(x) for x in paths}
    previous=shared.verified_snapshot(base,sources)
    scope_path=ROOT/'B/fixed-slo-scope-update-v1.json'
    p.need(current.sha(scope_path)==SCOPE_SHA,'fixed-SLO scope correction changed')
    scope_ref=p.ref(scope_path);scope=p.checked(scope_ref);sources[str(scope_path)]=scope_ref['sha256']
    p.need(scope['schema']=='B-fixed-SLO-current-scope-v1'
        and scope['historical_scale11_required_for_this_task'] is False
        and scope['historical_scale11_unmeasured'] is True
        and scope['historical_observed_archive_rows']==259,'wrong historical scope')
    for key in ('original_historical_declaration','actual_no_GPU_or_waiter'):
        reference=scope[key];p.checked(reference);sources[reference['path']]=reference['sha256']
    points=copy.deepcopy(previous['points'])
    missing=[x for x in points if x['phase']=='scale' and not x['metrics_verified']]
    p.need(len(missing)==11 and {x['cell_id'] for x in missing}=={x['cell_id'] for x in scope['excluded_historical_cells']},
        'historical scope does not match exactly the eleven existing gaps')
    p.need(sum(x['phase']=='main' and x['metrics_verified'] for x in points)==450
        and sum(x['phase']=='scale' and x['metrics_verified'] for x in points)==259,'existing raw coverage changed')
    overlay=eligibility.verify(points,p.ref(ROOT/'C/all-model-original-Dynamo90-arrival-audit-v2.json'),sources)
    view=eligibility.comparison_view(points,overlay)
    p.need(points==previous['points'],'historical records were changed')
    document=copy.deepcopy(previous)
    document.update(created_s=time.time(),existing_history_archive_complete=True,
        new_historical_measurements_required=False,fixed_slo_scope=scope_ref,
        historical_scale_declared=270,historical_scale_observed=259,historical_scale_unmeasured=11,
        historical_unmeasured_not_required_by_current_scope=[x['cell_id'] for x in missing],
        original_450_exactly_unchanged=True,original_259_scale_exactly_unchanged=True,
        all_original_720_records_exactly_unchanged=True,historical_scientific_overlay_applied=True,
        scientific_comparison_overlay=overlay,
        previous_snapshot=dict(path=str(base),sha256=current.sha(base/'manifest.json')))
    old=shared.load_historical();out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    old.write_csv(out/'points.csv',points)
    old.write_csv(out/'scientific-comparison-points.csv',view)
    old.write_csv(out/'engineering-exclusions.csv',overlay['quarantined'])
    old.write_csv(out/'historical-unmeasured-outside-current-scope.csv',missing)
    old.write_csv(out/'paired-comparisons.csv',old.pairwise(view))
    old.write_csv(out/'scale-display.csv',old.scale_display(view))
    old.write_csv(out/'request-counts.csv',document['request_counts'])
    (out/'scientific-comparison-overlay.json').write_text(json.dumps(overlay,indent=2)+'\n')
    old.figures(view,out)
    (out/'README.md').write_text('# 已有历史数据归档\n\n保留原450个固定SLO主表观测及259个已测SLO-scale观测，原720条声明记录逐字段不变。另11个历史EcoServe SLO-scale点仍未测；按用户最新要求，本轮只在固定SLO下增加rate，因此这11点不再补测，也不计入本轮缺口。\n\n独立修正表隔离19个有逐请求证据的历史工程缺陷：2个32B DynamoLLM调度迟发，以及17个EcoServe已准入prefill的窗口关闭后未再开启。原能耗与计数完整保存在points.csv；只有派生科学比较与曲线排除这些点。所有其他低SLO及容量不足观测保留。原始记录核验数量不等于所有观测均可用于比较。\n')
    for path,h in sources.items():p.need(current.sha(path)==h,'historical evidence changed during archive')
    (out/'manifest.json').write_text(json.dumps(dict(sources=sources,
        files={str(x.relative_to(out)):current.sha(x) for x in out.rglob('*') if x.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(out=str(out),existing_main=450,existing_scale=259,unmeasured_scale=11,new_scale_measurements_required=False,quarantined_historical_cells=len(overlay['quarantined']),physical_host=__import__('socket').gethostname(),pid=__import__('os').getpid())))


if __name__=='__main__':main()
