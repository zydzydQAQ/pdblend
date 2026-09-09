"""Build the complete presentation archive after rate and historical coverage finish."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT=Path(__file__).resolve().parent
FIGURES={'.pdf','.png','.svg','.eps','.jpg','.jpeg'}
TABLES={'.csv','.tsv','.xlsx','.xls','.tex'}
RAW_NAMES={'bench.csv','power.csv','clocks.csv','frequency.csv','power_samples.csv','telemetry.csv','memory.csv'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def need(value,message):
    if not value:raise ValueError(message)


def check_report(folder):
    manifest=read(folder/'manifest.json')
    for relative,digest in manifest['files'].items():
        need(sha(folder/relative)==digest,'changed report artifact: '+str(folder/relative))
    return read(folder/'results.json')


def selected(path):
    if path.suffix.lower() in FIGURES:return True
    if path.suffix.lower() not in TABLES:return False
    return path.name not in RAW_NAMES and not {'raw','cells','operations','traces','datasets'}&set(path.relative_to(ROOT).parts)


def package(final_report,historical_complete,history,out):
    need(not out.exists(),'immutable delivery directory already exists')
    result=check_report(final_report);historical=check_report(historical_complete)
    need(result.get('final_report_complete') is True,'selected rate comparison still has required gaps')
    need(sum(x['verified_main'] for x in historical['models'])==450
         and sum(x['verified_scale'] for x in historical['models'])==270,'historical 450+270 coverage incomplete')
    need(historical.get('historical_scientific_overlay_applied') is True
         and historical['scientific_comparison_overlay']['applies_to_comparisons_and_figures'] is True,
         'historical scientific correction has not been applied to paired comparisons and figures')
    history_manifest=read(history/'manifest.json')
    old_zip=history/'historical-tables-and-figures.zip'
    need(sha(old_zip)==history_manifest['files'][old_zip.name],'historical archive changed')
    candidates={path for path in ROOT.rglob('*') if path.is_file() and selected(path)}
    for parent in {p.parent for p in candidates}:
        for name in ('REPORT.md','README.md','report.md','manifest.json','results.json','boundaries.json','adaptive-next-plan.json'):
            path=parent/name
            if path.is_file():candidates.add(path)
    for folder in (final_report,historical_complete):
        candidates.update(p for p in folder.rglob('*') if p.is_file())
    old_index=list(csv.DictReader((history/'artifact-index.csv').open()))
    expected_old={row['archive_path']:row for row in old_index}
    rows=[];members=set();out.mkdir(parents=True)
    archive=out/'all-tables-and-figures.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        with zipfile.ZipFile(old_zip) as previous:
            need(set(previous.namelist())==set(expected_old),'historical archive index differs')
            for name in previous.namelist():
                raw=previous.read(name);old=expected_old[name]
                need(hashlib.sha256(raw).hexdigest()==old['sha256'],'historical member changed')
                member='historical/'+name;need(member not in members,'duplicate archive member');members.add(member)
                target.writestr(member,raw)
                rows.append(dict(section='historical',source_path=old['source_path'],archive_path=member,
                    sha256=old['sha256'],bytes=len(raw),kind=old['kind']))
        for path in sorted(candidates):
            before=path.stat();raw=path.read_bytes();after=path.stat()
            need(before.st_mtime_ns==after.st_mtime_ns and before.st_size==after.st_size,'artifact is still being written: '+str(path))
            member='current/'+str(path.relative_to(ROOT));need(member not in members,'duplicate current member');members.add(member)
            target.writestr(member,raw)
            rows.append(dict(section='current',source_path=str(path),archive_path=member,
                sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                kind='figure' if path.suffix.lower() in FIGURES else 'table' if path.suffix.lower() in TABLES else 'provenance'))
        overview=('全部历史与本轮表格、figure归档\n\n'
            +'最终 rate 比较：current/'+str(final_report.relative_to(ROOT))+'/REPORT.md\n'
            +'历史 450 主表 +270 SLO-scale：current/'+str(historical_complete.relative_to(ROOT))+'/README.md\n\n'
            +'所有历史版本和阶段快照分别保留。文件数量不是独立实验数量。'
            +'当前 rate 图以到达率为横轴，重复与实际源码/画像/策略分开；未完成请求、低 SLO、未测点不删除。'
            +'90%只决定停止上探；开发比较与严格双方服务/节能标签分别报告。'
            +'超时缺少最终用量时，记录的 token 总数可能是下界；文本块不冒充 token。'
            +'历史原表保留原始数值；已诊断的调度卡顿点另有科学资格修正表，最终比较和图排除其胜负，旧快照可能尚未包含此修正。'
            +'全部八卡能耗包含实际排空和转换，重叠窗口不相加。原始逐请求、功率、频率数据仍在索引对应的工作区来源。\n')
        target.writestr('INDEX.txt',overview)
    with (out/'artifact-index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'README.md').write_text(overview)
    with zipfile.ZipFile(archive) as packed:
        need(packed.testzip() is None,'archive CRC verification failed')
        need(set(packed.namelist())==members|{'INDEX.txt'},'final archive member set differs')
    for row in rows:
        if row['section']=='current':need(sha(row['source_path'])==row['sha256'],'current source changed during packaging')
    (out/'manifest.json').write_text(json.dumps(dict(created_s=time.time(),
        code_sha256=sha(__file__),final_report=dict(path=str(final_report),sha256=sha(final_report/'manifest.json')),
        historical_complete=dict(path=str(historical_complete),sha256=sha(historical_complete/'manifest.json')),
        old_archive=dict(path=str(old_zip),sha256=sha(old_zip)),artifacts=len(rows),
        kinds=dict(Counter(row['kind'] for row in rows)),archive_crc_verified=True,
        artifacts_are_not_independent_measurement_counts=True,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,artifacts=len(rows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final-report',type=Path,required=True)
    parser.add_argument('--historical-complete',type=Path,required=True)
    parser.add_argument('--history',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();package(a.final_report.resolve(),a.historical_complete.resolve(),a.history.resolve(),a.out.resolve())
