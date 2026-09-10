"""Report an explicit run manifest; never scan or promote historical results."""
import argparse
import csv
import json
from pathlib import Path

from .evidence import evaluate,sha256


FIELDS=('system','variant','dataset','load','seed','split','validity','completed','n_expected',
        'generated_tokens','energy_j','slo_attainment','ttft_avg_s','tpot_avg_s',
        'req_throughput','output_tok_throughput','goodput_rps','gpu_util',
        'token_itl_p50_s','token_itl_p95_s','token_itl_p99_s',
        'startup_seconds','reconfiguration_tail_s')
VERDICTS={'target_achieved':'达到目标','target_not_achieved':'未达到目标',
          'evidence_insufficient':'证据不足'}


def collect(manifest):
    paths=[Path(p).resolve() for p in manifest.get('summaries',[])]
    if len(paths)!=len(set(paths)): raise ValueError('duplicate run artifact')
    rows=[json.loads(path.read_text()) for path in paths]
    def document(key,default):
        return json.loads(Path(manifest[key]).read_text()) if manifest.get(key) else default
    verdict=evaluate([r for r in rows if r.get('split')=='formal'],
        document('expected_cells',[]),document('mechanisms',{}),document('freeze',{}))
    return dict(verdict=verdict,rows=rows,
        artifacts={str(path):sha256(path) for path in paths},
        scope='explicit manifest only; development/calibration rows cannot satisfy formal acceptance')


def write_report(result,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'points.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS,extrasaction='ignore')
        writer.writeheader();writer.writerows(result['rows'])
    verdict=result['verdict']
    lines=[f"结论：**{VERDICTS[verdict['verdict']]}**。",'',
           '只有显式列入清单的运行进入本报告；开发、校准和历史运行不能满足正式验收。','',
           '| 策略 | 数据集 / 负载 / seed | 完成数 | 8 卡能耗 J | joint SLO | TTFT s | TPOT s | req/s | goodput req/s | GPU 利用率 | 状态 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def number(row,key,percentage=False):
        value=row.get(key)
        return '—' if value is None else f'{value*100:.2f}%' if percentage else f'{value:.4g}'
    for row in result['rows']:
        lines.append('| '+' | '.join([str(row.get('variant') or row.get('system')),
            '/'.join(str(row.get(k,'—')) for k in ('dataset','load','seed')),
            f"{row.get('completed',0)}/{row.get('n_expected',0)}",number(row,'energy_j'),
            number(row,'slo_attainment',True),number(row,'ttft_avg_s'),number(row,'tpot_avg_s'),
            number(row,'req_throughput'),number(row,'goodput_rps'),number(row,'gpu_util',True),
            str(row.get('split'))+'/'+str(row.get('validity'))])+' |')
    lines.extend(['','正式验收缺口：',''])
    lines.extend('- '+reason for reason in verdict['reasons'])
    if not verdict['reasons']: lines.append('- 正式证据完整；能耗与逐点 SLO 区间见 report.json。')
    lines.extend(['','能耗包含所有 8 卡及运行中重配置；吞吐量使用真实请求跨度。',
                  '缺少完整配对组时，不把可用子集的节能幅度判为全面胜出。',
                  '启动/暖机、逐 token 间隔和输出吞吐量详见 points.csv 及各原始运行。',''])
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=collect(json.loads(args.manifest.read_text()))
    write_report(result,args.out)
    print(json.dumps(dict(verdict=result['verdict']['verdict'],runs=len(result['rows']))))


if __name__=='__main__': main()
