"""Append only the eleven declared EcoServe scale gaps; preserve all earlier points."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import collect_v13 as current
import inspect_historical_suffix_v1 as suffix
import historical_comparison_overlay_v1 as eligibility

ROOT=Path(__file__).resolve().parent
HISTORICAL_CODE=ROOT.parent/'five-system-results-v4/report.py'


def load_historical():
    spec=importlib.util.spec_from_file_location('original_scale_presentation',HISTORICAL_CODE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verified_snapshot(directory, sources):
    manifest=current.read(directory/'manifest.json')
    for relative, sha in manifest['files'].items():
        path=directory/relative
        if current.sha(path)!=sha:raise ValueError('historical snapshot changed: '+str(path))
        sources[str(path)]=sha
    for name, sha in manifest['sources'].items():
        path=Path(name)
        if current.sha(path)!=sha:
            sealed=directory/'invocation-observations'/(sha+'.json')
            if 'invocations' not in path.parts or not sealed.exists() or current.sha(sealed)!=sha:
                raise ValueError('historical raw evidence changed: '+name)
            sources[str(sealed)]=sha
        else:sources[name]=sha
    sources[str(directory/'manifest.json')]=current.sha(directory/'manifest.json')
    return current.read(directory/'results.json')


def append(base, declaration_path, execution_root, out, timing_audit, allow_partial=False, plots=False):
    if out.exists():raise FileExistsError(out)
    audit=current.load_audit();p=audit.p;old=load_historical()
    sources={str(x):current.sha(x) for x in [Path(__file__),Path(suffix.__file__),HISTORICAL_CODE,
        ROOT/'collect_v13.py',ROOT/'raw_metrics_v3.py',ROOT/'raw_metrics_v2.py',
        ROOT/'source_identity_v2.py',ROOT/'source_identity.py',ROOT/'final_selected_baseline_v1.py',
        Path(eligibility.__file__)]}
    previous=verified_snapshot(base,sources)
    declaration_ref=p.ref(declaration_path);declaration=p.checked(declaration_ref)
    sources[str(declaration_path)]=declaration_ref['sha256']
    p.need(declaration['schema']=='B-historical-ecoserve-scale11-until-complete-v1'
           and len(declaration['cells'])==11, 'wrong suffix declaration')
    original_manifest=p.read(old.DEFAULT_SOURCES['32b'][0])
    declared={r['cell_id']:r for r in original_manifest['cells']}
    points=copy.deepcopy(previous['points']);index={x['cell_id']:x for x in points}
    main_before=[copy.deepcopy(x) for x in points if x['phase']=='main']
    proofs=[]
    for row in declaration['cells']:
        cid=row['cell_id'];p.need(row==declared[cid], 'suffix changed original declared row')
        target=index[cid]
        p.need(target['phase']=='scale' and not target['metrics_verified'], 'suffix overlaps an existing observation')
        checkpoint=execution_root/'results/checkpoints'/(cid+'.json')
        value=suffix.inspect(audit,row,checkpoint if checkpoint.exists() else None,declaration_ref)
        proofs.append(value)
        if not value['measurement_valid']:
            if not allow_partial:raise ValueError('suffix not yet independently valid: '+cid+' '+str(value['error']))
            continue
        main=index[row['reuse_main_cell_id']]
        p.need(main['metrics_verified'], 'same-trace main reference is unverified')
        receipt=p.checked(value['receipt']);binding=p.checked(value['binding'])
        summary=receipt['summary']
        old.metric_observations(target,summary,receipt)
        target.update({k:v for k,v in value.items() if k not in ('status','phase','error','declaration')})
        target.update(status='completed',metrics_verified=True,checkpoint_verified=True,error=None,
            checkpoint_path=str(checkpoint),receipt_path=value['receipt']['path'],
            binding_sha256=value['binding']['sha256'],
            gpu_util_vector_valid=True,gpu_util_eight_board_consistent=True,
            metric_notes=[value['generated_tokens_semantics']] if not value.get('generated_token_count_complete',False) else [],
            executed_source=dict(binding_path=value['binding']['path'],binding_sha256=value['binding']['sha256'],
                host_release=binding['host_release'],host_manifest_sha256=value['host_manifest_sha256'],
                original_suffix_declaration=declaration_ref,producer_checkpoint=value['checkpoint'],
                actual_engine_identity=value['actual_engine_identity'],
                source_profile_policy_verified_against_original_binding=True),
            historical_suffix_verification=value['verification'])
        sources[str(checkpoint)]=current.sha(checkpoint)
        sources[value['receipt']['path']]=value['receipt']['sha256']
        sources[value['binding']['path']]=value['binding']['sha256']
        sources.update(p.read(checkpoint)['artifacts'])
    p.need([x for x in points if x['phase']=='main']==main_before, 'one of the original 450 main points changed')
    for original, now in zip(previous['points'],points):
        if original['metrics_verified']:p.need(original==now, 'previous verified historical observation changed')
    overlay=eligibility.verify(points,p.ref(timing_audit),sources)
    view=eligibility.comparison_view(points,overlay)
    document=copy.deepcopy(previous);document.update(created_s=time.time(),points=points,
        append_only_original_scale_suffix=True,original_450_exactly_unchanged=True,
        original_259_scale_exactly_unchanged=True,suffix_proofs=proofs,
        historical_scientific_overlay_applied=True,scientific_comparison_overlay=overlay,
        suffix_declaration=declaration_ref,previous_snapshot=dict(path=str(base),sha256=current.sha(base/'manifest.json')))
    for model in document['models']:
        group=[x for x in points if x['model']==model['model']]
        model.update(verified_main=sum(x['phase']=='main' and x['metrics_verified'] for x in group),
            verified_scale=sum(x['phase']=='scale' and x['metrics_verified'] for x in group),
            valid_below_90=sum(x['metrics_verified'] and x['slo_attainment']<.9 for x in group),
            point_errors=[dict(cell_id=x['cell_id'],error=x['error']) for x in group if x['error']])
    out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    old.write_csv(out/'points.csv',points)
    old.write_csv(out/'scientific-comparison-points.csv',view)
    old.write_csv(out/'engineering-exclusions.csv',overlay['quarantined'])
    (out/'scientific-comparison-overlay.json').write_text(json.dumps(overlay,indent=2)+'\n')
    old.write_csv(out/'paired-comparisons.csv',old.pairwise(view))
    old.write_csv(out/'scale-display.csv',old.scale_display(view))
    old.write_csv(out/'new-suffix-points.csv',proofs)
    old.write_csv(out/'request-counts.csv',previous['request_counts'])
    if plots:old.figures(view,out)
    (out/'README.md').write_text('历史表追加原先缺少的 11 个 EcoServe 32B SLO-scale 点；原 450 个主表点与已核验的 259 个 scale 点逐字段保持不变。新点沿用原 trace、原 SLO 和原控制策略，独立核验工作量与八卡能耗。所有未完成工作与低 SLO 负结果保留。\n\n原 32B DynamoLLM Alpaca rate=4、ShareGPT rate=2 的调度卡顿改变了到达时序，已在独立修正表中标记为科学比较无效。原始计数与能耗仍完整保存在 points.csv；paired-comparisons.csv 和 figure 使用修正后的资格，不计这两点的胜负。450 个原始记录的核验数量不代表 450 个点均可用于科学比较。\n')
    for path,sha in sources.items():p.need(current.sha(path)==sha,'evidence changed during appendix: '+path)
    (out/'manifest.json').write_text(json.dumps(dict(sources=sources,
        files={str(x.relative_to(out)):current.sha(x) for x in out.rglob('*') if x.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(models=document['models'],suffix_verified=sum(x['measurement_valid'] for x in proofs))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--declaration',type=Path,required=True);parser.add_argument('--execution-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--timing-audit',type=Path,required=True)
    parser.add_argument('--plots',action='store_true');args=parser.parse_args()
    append(args.base.resolve(),args.declaration.resolve(),args.execution_root.resolve(),args.out.resolve(),args.timing_audit.resolve(),args.allow_partial,args.plots)
