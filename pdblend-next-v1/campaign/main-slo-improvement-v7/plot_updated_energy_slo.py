"""Two reproducible figures for one frozen development version, never a splice."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import protocol as p


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib import font_manager
    p.need(not args.out.exists(), 'new figure snapshot required')
    result=p.read(args.report/'results.json');manifest=p.read(args.report/'manifest.json')
    p.need(all(p.sha(args.report/name)==digest for name,digest in manifest['files'].items()),
           'verified result snapshot changed')
    points=[v for v in result['results'] if v['measurement_valid']]
    p.need(len(points)==result['verified'] and all(v['arm']=='fixed2' for v in points),
           'this snapshot contains only the current fixed-two version')
    originals=p.original_points();by_pair={p.pair_identity(v)+(v['system'],):v for v in originals}
    snapshots=[p.read(args.snapshots/n/'snapshot.json') for n in ('A','B','C')]
    stamps=[v['captured_s'] for v in snapshots]
    interval='–'.join(datetime.fromtimestamp(v,ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S') for v in (min(stamps),max(stamps)))
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
        'font.size':10,'axes.titlesize':12,'pdf.fonttype':42,'ps.fonttype':42})
    args.out.mkdir(parents=True)
    styles={'pdblend':('#D84A40','P','PDBlend v7（固定两实例）'),
            'mixed':('#D99117','s','Mixed'),'distserve':('#278568','^','DistServe'),
            'dynamollm':('#9865B3','D','DynamoLLM*'),'ecoserve':('#687984','v','EcoServe')}
    metrics=[('energy_j',.001,'总八卡 GPU 能耗（kJ）','能耗'),
             ('slo_attainment',100.,'Joint SLO attainment（%）','SLO attainment')]
    checks=[]
    with PdfPages(args.out/'energy-and-slo.pdf') as combined:
        for metric,scale,ylabel,title in metrics:
            fig,axes=plt.subplots(3,3,figsize=(16.5,11.6),squeeze=False)
            new_count=baseline_count=0
            for i,model in enumerate(p.MODELS):
                for j,dataset in enumerate(p.DATASETS):
                    ax=axes[i,j]
                    for system in p.BASELINES:
                        rows=sorted([v for v in originals if v['model']==model and v['dataset']==dataset
                            and v['system']==system],key=lambda v:v['rate_rps'])
                        p.need(len(rows)==10,'complete frozen baseline rate grid required')
                        color,marker,_=styles[system]
                        ax.plot([v['rate_rps'] for v in rows],[v[metric]*scale for v in rows],
                            color=color,marker=marker,ms=4,lw=1.25,alpha=.85,zorder=2)
                        bad=[v for v in rows if v.get('work_complete') is not True]
                        ax.scatter([v['rate_rps'] for v in bad],[v[metric]*scale for v in bad],
                            marker='x',s=40,color='black',lw=1,zorder=5)
                        baseline_count+=len(rows)
                    selected=[v for v in points if v['model']==model and v['dataset']==dataset]
                    for repeat in sorted({v['repeat'] for v in selected}):
                        rows=sorted([v for v in selected if v['repeat']==repeat],key=lambda v:v['rate_rps'])
                        color,marker,_=styles['pdblend']
                        ax.plot([v['rate_rps'] for v in rows],[v[metric]*scale for v in rows],
                            color=color,marker=marker,ms=8,lw=2.4,ls='-' if repeat==1 else '--',zorder=6)
                        bad=[v for v in rows if not v['work_complete']]
                        ax.scatter([v['rate_rps'] for v in bad],[v[metric]*scale for v in bad],
                            marker='x',s=85,color='black',lw=1.3,zorder=7)
                        new_count+=len(rows)
                    n_rates=len({v['rate_rps'] for v in selected})
                    ax.text(.025,.045,f'v7 已测 {n_rates}/10 个 rate' if selected else 'v7 尚无完成测量',
                        transform=ax.transAxes,fontsize=9,color='#992D29',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=2))
                    if metric=='slo_attainment':
                        ax.axhline(90,color='#343434',ls=':',lw=1)
                        ax.set_ylim(-2,103)
                    else:
                        ax.set_ylim(bottom=0)
                    ax.set_title(f'{model.upper()} · {dataset.title() if dataset!="longbench" else "LongBench"}')
                    ax.set_xlabel('Rate（req/s，线性刻度）');ax.set_ylabel(ylabel)
                    ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
            handles=[Line2D([],[],color=c,marker=m,lw=2.4 if k=='pdblend' else 1.4,
                markersize=7 if k=='pdblend' else 5,label=label) for k,(c,m,label) in styles.items()]
            fig.suptitle(f'PDBlend v7 与四个 baseline：{title}',y=.985,fontsize=17)
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=5,frameon=False)
            fig.text(.5,.023,f'2026-09-08 北京时间 {interval} 快照；v7 共 {len(points)} 次完成测量，全部为首次重复。',ha='center',fontsize=10)
            fig.text(.5,.007,'×：规定工作未全部完成。PDBlend 未测 rate 留空，不用旧版本补线。基线为 snapshot-006；*7B/32B DynamoLLM 为 resident。',ha='center',fontsize=9)
            fig.tight_layout(rect=(.005,.047,.995,.92))
            fig.savefig(args.out/(metric+'.png'),dpi=180)
            fig.savefig(args.out/(metric+'.pdf'))
            combined.savefig(fig);plt.close(fig)
            checks.append(dict(metric=metric,plotted_pdblend_observations=new_count,
                plotted_baseline_observations=baseline_count,expected_pdblend=len(points),
                expected_baseline=360,passed=new_count==len(points) and baseline_count==360))
    table=[]
    for row in sorted(points,key=lambda v:(p.MODELS.index(v['model']),p.DATASETS.index(v['dataset']),v['rate_rps'],v['repeat'])):
        value={k:row[k] for k in ('model','dataset','rate_rps','repeat','cell_id','implementation_id',
            'energy_j','slo_attainment','completed_work_requests','n_expected','work_complete','measurement_valid')}
        value['energy_kj']=row['energy_j']/1000;value['slo_percent']=row['slo_attainment']*100
        for system in p.BASELINES:
            old=by_pair[p.pair_identity(row)+(system,)];verdict=p.verdict(row,old)
            value.update({system+'_energy_kj':old['energy_j']/1000,system+'_slo_percent':old['slo_attainment']*100,
                system+'_energy_reduction_percent':-verdict['energy_change_pct'],system+'_slo_required_percent':verdict['slo_required']*100,
                system+'_energy_pass':verdict['energy_pass'],system+'_slo_pass':verdict['slo_pass'],system+'_joint_pass':verdict['passed']})
        table.append(value)
    with (args.out/'updated-cells.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    shutil.copyfile(args.report/'points.csv',args.out/'all-declared-status.csv')
    shutil.copyfile(args.report/'paired-baselines.csv',args.out/'paired-baselines.csv')
    lines=['# 当前 PDBlend 更新负载格子与能耗、SLO 曲线','',
        f'快照时间：2026-09-08 北京时间 {interval}。只展示当前统一 v7 固定两实例策略的 {len(points)} 个已完成测量格子；全部是首次重复，第二次尚未纳入。',
        '', '原450点已采齐。此处是新策略在原问题负载上的独立验证，四个baseline沿用snapshot-006。旧v4/v5/v6结果独立保留，不拼接不同版本的最好点。',
        '', '| 模型 | Alpaca 已测 rate | ShareGPT 已测 rate | LongBench 已测 rate | 总格子 |',
        '|---|---|---|---|---:|']
    for model in p.MODELS:
        cells=[v for v in points if v['model']==model]
        loads=[', '.join(f'{v:g}' for v in sorted({x['rate_rps'] for x in cells if x['dataset']==ds})) or '尚未完成' for ds in p.DATASETS]
        lines.append('| '+model.upper()+' | '+' | '.join(loads)+f' | {len(cells)} |')
    pairs=[v for v in result['pairs'] if v['measurement_valid']]
    lines += ['',f'这 {len(points)} 个测点对应 {len(pairs)} 组 baseline 配对，能耗有 {sum(v["energy_pass"] for v in pairs)}/{len(pairs)} 组不高于 baseline；SLO 有 {sum(v["slo_pass"] for v in pairs)}/{len(pairs)} 组满足 min(90%, baseline SLO)。',
        '',f'但只有 {sum(v["work_complete"] for v in points)}/{len(points)} 个测点全部完成规定工作，联合验收通过 {sum(v["passed"] for v in pairs)}/{len(pairs)} 组。因此目前不能判定新版已满足你的整体优越标准。请求未完成时，低总能耗尤其不能单独解释为节能收益。',
        '', '| 模型 | 数据集 | rate | 能耗 kJ | SLO % | 完成请求/规定请求 | 联合通过 baseline 数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for v in table:
        lines.append(f'| {v["model"].upper()} | {v["dataset"]} | {v["rate_rps"]:g} | {v["energy_kj"]:.3f} | {v["slo_percent"]:.2f} | {v["completed_work_requests"]}/{v["n_expected"]} | '+str(sum(v[s+'_joint_pass'] for s in p.BASELINES))+'/4 |')
    lines += ['', '图中保留全部已核验点和不完整工作标记。红线只连接同一版本、同一次重复的实测点；不填入未测rate、不平滑、不裁去极端值。总能耗是主测窗口内全部八卡能耗，重叠的外层窗口不相加；部署准备能耗另见来源报告。',
        '', '7B LongBench rate=3、14B Alpaca rate=12 与 LongBench rate=1.25、32B ShareGPT rate=2 仍有明显低SLO。多个其余格子SLO较高，但仍因少量请求失败而不满足完整工作要求。',
        '', '单种子重复不提供独立到达种子的置信区间。这一快照不包含动态实例主实验或900秒开发轨迹。']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.out/'plot-generator.py').write_bytes(Path(__file__).read_bytes())
    p.write(args.out/'validation.json',dict(checks=checks,snapshot_times_s=stamps,
        work_complete_count=sum(v['work_complete'] for v in points),point_count=len(points),
        no_cross_version_splicing=True,source_report=p.ref(args.report/'manifest.json')),exclusive=True)
    p.write(args.out/'manifest.json',dict(schema='energy-slo-plot-snapshot-v1',
        files={v.name:p.sha(v) for v in args.out.iterdir() if v.is_file()},source_report=p.ref(args.report/'manifest.json'),
        snapshot_manifests=[p.ref(args.snapshots/n/'snapshot.json') for n in ('A','B','C')],
        original_snapshot=p.PINNED),exclusive=True)
    print(json.dumps(dict(out=str(args.out.resolve()),points=len(points),checks=checks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report','snapshots','out'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
