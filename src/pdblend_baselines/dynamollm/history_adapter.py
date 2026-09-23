"""Stream pinned author arrival CSVs into auditable independent history.

The production timestamps and token counts are separate from the three prompt
corpora. Counts aggregate actual rows; they never synthesize repeated arrivals.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

from .policy import SHAPES,WeeklyLoadTemplate,classify


DAY_US=86400*1000000
FIELDS=['TIMESTAMP','ContextTokens','GeneratedTokens']
PINNED_ASSETS={
    'AzureLLMInferenceTrace_code_1week.csv':'71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f',
    'AzureLLMInferenceTrace_conv_1week.csv':'a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941',
}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def timestamp_us(text):
    try:
        date=datetime.fromisoformat(text)
        if date.tzinfo is None or date.utcoffset() is None:raise ValueError('explicit timezone required')
        difference=date.astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
        return (difference.days*86400+difference.seconds)*1000000+difference.microseconds
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('invalid explicit-timezone arrival timestamp') from exc


def prepare_trace(path,output,*,expected_sha256,holdout_seconds=1890,provenance=None):
    """Prepare one service independently; insufficient holdout is reported.

    Week exposure uses whole UTC dates in the publisher's one-week trace.
    Observed first/last timestamps are reported separately, including the
    unobserved edges; no invocation is added at these date boundaries.
    """
    path=Path(path).resolve();output=Path(output).resolve()
    if not path.is_file() or sha(path)!=expected_sha256:raise ValueError('arrival source SHA mismatch')
    if type(holdout_seconds) is not int or holdout_seconds<1890:raise ValueError('original-cycle holdout requires at least 1890 seconds')
    output.mkdir(parents=True,exist_ok=False)
    implementation=output/'implementation';implementation.mkdir()
    source_files={}
    for name in ('history_adapter.py','policy.py'):
        current=Path(__file__).with_name(name);source_files[name]=sha(current)
        shutil.copy2(current,implementation/name)
    bins={};rows=0;previous=None;first=None;last=None;days=Counter();shapes=Counter();zeros=Counter();limits=Counter()
    lengths={k:dict(min=None,max=0,total=0) for k in ('input_tokens','output_tokens')}
    pending=output/'heldout-arrivals.pending.jsonl';holdout_rows=0
    with path.open(newline='',encoding='utf-8-sig') as handle,pending.open('w') as holdout:
        reader=csv.reader(handle)
        if next(reader,None)!=FIELDS:raise ValueError('official arrival CSV schema differs')
        for line,row in enumerate(reader,2):
            if len(row)!=3:raise ValueError('arrival CSV row width differs at line '+str(line))
            at=timestamp_us(row[0]);n,o=int(row[1]),int(row[2])
            if n<0 or o<0:raise ValueError('negative author token count at line '+str(line))
            if previous is not None and at<previous:raise ValueError('arrival rows must be chronologically ordered')
            if first is None:first=at;calendar_start=at//DAY_US*DAY_US
            previous=last=at;rows+=1;shape=classify(n,o);day=at//DAY_US;slot=at//(300*1000000)
            days[day]+=1;shapes[shape]+=1
            for name,value in (('input_tokens',n),('output_tokens',o)):
                stats=lengths[name];stats['min']=value if stats['min'] is None else min(stats['min'],value)
                stats['max']=max(stats['max'],value);stats['total']+=value
                if value==0:zeros[name]+=1
            if n>7168:limits['input_above_7168']+=1
            if o>512:limits['output_above_512']+=1
            if n+o>8192:limits['context_above_8192']+=1
            key=(slot,shape)
            if key not in bins:
                bins[key]=dict(at_s=at/1000000,input_tokens=n,output_tokens=o,count=0,
                    record_type='aggregated_verified_arrivals',source_row_first=rows,source_row_last=rows,
                    source_timestamp_first=row[0],source_timestamp_last=row[0],split='calibration')
            record=bins[key];record['count']+=1;record['source_row_last']=rows;record['source_timestamp_last']=row[0]
            holdout_start=calendar_start+WeeklyLoadTemplate.WEEK*1000000
            if holdout_start<=at<holdout_start+holdout_seconds*1000000:
                holdout.write(json.dumps(dict(source_row=rows,source_timestamp=row[0],source_timestamp_us=at,
                    offset_s=(at-holdout_start)/1000000,split='chronological_holdout'),separators=(',',':'))+'\n')
                holdout_rows+=1
            if rows%1000000==0:
                save(output/'progress.json',dict(rows=rows,last_timestamp=row[0],source_sha256=expected_sha256))
    if first is None:raise ValueError('arrival trace has no actual requests')
    calendar_end=(last//DAY_US+1)*DAY_US;week_us=WeeklyLoadTemplate.WEEK*1000000
    usable=calendar_end-calendar_start>=week_us
    records=[value for key,value in sorted(bins.items())]
    base=dict(schema=2,split='calibration',evaluation_trace_used=False,source_path=str(path),source_sha256=expected_sha256,
        source_lengths='author ContextTokens/GeneratedTokens; original production shapes',
        prompt_content_available=False,prompt_corpora_used=False,timestamp_replay_speed=1,
        slot_s=300,boundary_basis='whole UTC dates covered by publisher trace; no invented invocations',
        observed_first_timestamp_us=first,observed_last_timestamp_us=last,provenance=provenance or {},
        temporal_endpoint_assumption='publisher collection covers UTC dates containing first and last recorded invocation',
        formal_eligible=False,hardware_qualified=False)
    if usable:
        start=(calendar_end-week_us)/1000000;end=calendar_end/1000000
        selected=[r for r in records if start<=r['at_s']<end]
        value=dict(base,start_s=start,end_s=end,records=selected,aggregated_requests=sum(r['count'] for r in selected),
            historical_week_policy='most recent complete calendar week of this one service')
        WeeklyLoadTemplate.fit(selected,start_s=start,end_s=end)
        save(output/'history.json',value)
    available=usable and last>=holdout_start+holdout_seconds*1000000 and holdout_rows>0
    holdout=dict(available=available,seconds=holdout_seconds,start_s=holdout_start/1000000,
        end_s=holdout_start/1000000+holdout_seconds,actual_rows=holdout_rows,
        purpose='optional chronological weekly-prediction holdout; separate from 100-second rate-scale and topology functional contracts')
    if available:
        start=calendar_start/1000000;end=holdout_start/1000000
        selected=[r for r in records if start<=r['at_s']<end]
        save(output/'chronological-training.json',dict(base,start_s=start,end_s=end,records=selected,
            aggregated_requests=sum(r['count'] for r in selected),historical_week_policy='first complete calendar week before holdout'))
        pending.rename(output/'heldout-arrivals.jsonl')
        holdout['arrivals_sha256']=sha(output/'heldout-arrivals.jsonl')
    else:
        pending.rename(output/'partial-heldout-arrivals.jsonl')
        holdout['reason']='same-service trace lacks a complete first-week plus later observed holdout; no loop or cross-service stitching'
    daily={datetime.fromtimestamp(day*86400,timezone.utc).date().isoformat():count for day,count in sorted(days.items())}
    if sha(path)!=expected_sha256 or any(sha(Path(__file__).with_name(name))!=value for name,value in source_files.items()):
        raise ValueError('arrival source or adapter code changed during preparation')
    summary=dict(source_path=str(path),source_sha256=expected_sha256,source_bytes=path.stat().st_size,rows=rows,
        source_observed_duration_s=(last-first)/1000000,calendar_window_s=(calendar_end-calendar_start)/1000000,
        first_timestamp_us=first,last_timestamp_us=last,rows_per_utc_day=daily,shape_counts=dict(shapes),
        zero_token_rows=dict(zeros),outside_7b_workload_limits=dict(limits),lengths=lengths,
        usable_independent_history=usable,chronological_holdout=holdout,
        historical_week_may_be_used_for_independent_prompt_corpus_experiments=usable,
        prediction_accuracy_on_production_holdout_measured=False,rate_scale_arrivals='unchanged seed701/Poisson 100 seconds',
        preprocessing_sha256=source_files['history_adapter.py'],implementation_files=source_files,
        source_and_implementation_unchanged=True,formal_eligible=False)
    summary['artifacts']={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    save(output/'summary.json',summary)
    return summary


def prepare_official(manifest_path,project):
    manifest_path=Path(manifest_path).resolve();source=json.loads(manifest_path.read_text())
    if (source.get('complete') is not True or source.get('repository')!='Azure/AzurePublicDataset'
            or source.get('revision')!='207bed67dd10090b28ad4f745b2cfd41a11aace4'
            or source.get('tag')!='dataset-llm-2024'):
        raise ValueError('complete pinned official author download manifest required')
    receipts={a['name']:a for a in source.get('assets',[])}
    if set(receipts)!=set(PINNED_ASSETS):raise ValueError('both separately identified author services required')
    references={}
    for name,expected in source['source_files'].items():
        path=manifest_path.parent/name
        if path.name!=name or not path.is_file() or sha(path)!=expected:raise ValueError('pinned official reference changed')
        references[str(path)]=expected
    output=Path(project)/'results'/time.strftime('%Y-%m-%d')/(time.strftime('%H%M%S')+'-dynamo-author-history-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False);print(str(output),flush=True);results={}
    for name,expected in PINNED_ASSETS.items():
        receipt=receipts[name]
        if receipt.get('verified') is not True or receipt.get('sha256')!=expected:raise ValueError('pinned author asset receipt differs')
        service='code' if '_code_' in name else 'conversation'
        provenance=dict(download_manifest=str(manifest_path),download_manifest_sha256=sha(manifest_path),
            official_reference_files=references,repository=source['repository'],revision=source['revision'],service=service,
            attribution='Stojkovic et al., DynamoLLM, HPCA 2025; AzurePublicDataset, CC-BY')
        results[service]=prepare_trace(receipt['path'],output/service,expected_sha256=expected,provenance=provenance)
    save(output/'manifest.json',dict(services=results,source_manifest_sha256=sha(manifest_path),
        service_timelines_combined=False,formal_eligible=False))
    return output


def prepare_verified_asset(receipt_path,project):
    """Process a completely verified service while another asset downloads."""
    receipt_path=Path(receipt_path).resolve();receipt=json.loads(receipt_path.read_text())
    name=receipt.get('name');expected=PINNED_ASSETS.get(name)
    if expected is None or receipt.get('verified') is not True or receipt.get('sha256')!=expected:
        raise ValueError('verified pinned individual author asset required')
    original=json.loads((receipt_path.parent/'manifest.json').read_text())
    if (original.get('repository')!='Azure/AzurePublicDataset'
            or original.get('revision')!='207bed67dd10090b28ad4f745b2cfd41a11aace4'):
        raise ValueError('pinned author reference revision required')
    references={}
    for relative,value in original['source_files'].items():
        path=receipt_path.parent/relative
        if path.name!=relative or sha(path)!=value:raise ValueError('pinned author reference changed')
        references[str(path)]=value
    service='code' if '_code_' in name else 'conversation'
    output=Path(project)/'results'/time.strftime('%Y-%m-%d')/(time.strftime('%H%M%S')+'-dynamo-author-'+service+'-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False);print(str(output),flush=True)
    save(output/'official-reference-snapshot.json',original)
    provenance=dict(download_receipt=str(receipt_path),download_receipt_sha256=sha(receipt_path),
        reference_snapshot_sha256=sha(output/'official-reference-snapshot.json'),official_reference_files=references,
        repository=original['repository'],revision=original['revision'],service=service,
        attribution='Stojkovic et al., DynamoLLM, HPCA 2025; AzurePublicDataset, CC-BY')
    result=prepare_trace(receipt['path'],output/service,expected_sha256=expected,provenance=provenance)
    save(output/'manifest.json',dict(service=service,service_complete=True,service_timelines_combined=False,
        summary_sha256=sha(output/service/'summary.json'),result=result,formal_eligible=False))
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-manifest',type=Path)
    source.add_argument('--asset-receipt',type=Path)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args(argv)
    if args.asset_receipt:prepare_verified_asset(args.asset_receipt,args.project)
    else:prepare_official(args.source_manifest,args.project)


if __name__=='__main__':main()
