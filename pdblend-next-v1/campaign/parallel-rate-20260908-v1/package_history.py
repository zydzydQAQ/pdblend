"""Snapshot all pre-existing table/figure artifacts, preserving their versions.

This is an artifact catalogue, never a count of unique GPU experiments.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT=Path(__file__).resolve().parent
SOURCES=(Path('/root/workspace/pdblend-next-v1/campaign'),Path('/root/workspace/pdblend/new-results'))
RAW_CSV={'bench.csv','power.csv','clocks.csv','frequency.csv','power_samples.csv','telemetry.csv'}
FIGURES={'.pdf','.png','.svg','.eps','.jpg','.jpeg'}
TABLES={'.xlsx','.xls','.tex','.tsv'}

def selected(path):
    if ROOT==path or ROOT in path.parents:return False
    if path.suffix.lower() in FIGURES|TABLES:return True
    return path.suffix.lower()=='.csv' and path.name not in RAW_CSV and not {'traces','datasets'}&set(path.parts)

def package(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    candidates={}
    for source in SOURCES:
        for path in source.rglob('*'):
            if path.is_file() and selected(path):
                relative=Path(source.parent.name)/source.name/path.relative_to(source)
                candidates[path]=relative
    # Keep human descriptions and provenance adjacent to the scientific outputs.
    for parent in {p.parent for p in candidates}:
        for name in ('REPORT.md','README.md','report.md','manifest.json'):
            path=parent/name
            if path.is_file():
                source=next(s for s in SOURCES if s in path.parents)
                candidates[path]=Path(source.parent.name)/source.name/path.relative_to(source)
    rows=[];unstable=[];unique={};archive=out/'historical-tables-and-figures.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,relative in sorted(candidates.items()):
            before=path.stat();raw=path.read_bytes();after=path.stat()
            if before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size:
                unstable.append(str(path));continue
            digest=hashlib.sha256(raw).hexdigest()
            duplicate=unique.get(digest)
            if duplicate is None:unique[digest]=str(relative)
            z.writestr(str(relative),raw)
            rows.append(dict(source_path=str(path),archive_path=str(relative),sha256=digest,
                bytes=len(raw),extension=path.suffix,mtime_s=after.st_mtime,
                same_content_as=duplicate or '',
                kind='figure' if path.suffix.lower() in FIGURES else
                     'table' if path.suffix.lower() in TABLES|{'.csv'} else 'provenance',
                measurement_count='not_applicable_artifact_catalogue'))
    with (out/'artifact-index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata=dict(schema=1,created_s=time.time(),source_roots=list(map(str,SOURCES)),
        artifacts=len(rows),unique_contents=len(unique),kinds=dict(Counter(r['kind'] for r in rows)),
        uncompressed_bytes=sum(r['bytes'] for r in rows),archive_bytes=archive.stat().st_size,
        unstable_omitted=unstable,excluded_raw_csv_names=sorted(RAW_CSV),
        excluded_trace_directories=['traces','datasets'],
        historical_versions_separate=True,new_campaign_packaged_separately=True,
        artifacts_are_not_unique_experiment_counts=True,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('历史表格与figure原样归档，保留两个实验目录下的版本路径。\n\n'
        'artifact-index.csv记录每个文件的来源、SHA256及同内容副本；文件数量不是独立实验数量。'
        '历史图可能采用当时的筛选口径或横轴，保留供追溯；本轮另交付以rate为横轴的新统计图。'
        '原始逐请求、功率和频率采样文件仍在原路径，未混入表格压缩包。\n')
    print(json.dumps({k:metadata[k] for k in ('artifacts','unique_contents','kinds','archive_bytes','unstable_omitted')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    package(p.parse_args().out.resolve())
