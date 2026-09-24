#!/usr/bin/env python3
"""Measure raw logical journal growth before gzip, without GPUs."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

from pdblend.results.journal import CompactJournal, iter_journal


def observations(count):
    for i in range(count):
        text='token '*(i+1)
        payload=dict(request_id='r',token_ids=[100+i],token_index=i+1,text=text,
                     choices=[dict(index=0,text=text,finish_reason='length' if i==count-1 else None)],
                     at_s=100+i/20,finished=i==count-1,usage=dict(completion_tokens=i+1))
        yield dict(kind='eco_native_sse',request_id='r',at_s=100+i/20,payload=payload)
        yield dict(kind='eco_client_sse',request_id='r',at_s=100.01+i/20,payload=payload)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv);results=[]
    with tempfile.TemporaryDirectory(prefix='pdblend-compact-') as directory:
        for count in (128,256,512,1024):
            source=list(observations(count));path=Path(directory)/f'{count}.jsonl.gz'
            old_bytes=sum(len(json.dumps(row,sort_keys=True,separators=(',',':')).encode())+1 for row in source)
            with CompactJournal(path) as writer:
                for row in source:writer.write(row)
            with gzip.open(path,'rb') as handle:logical_bytes=len(handle.read())
            if list(iter_journal(path))!=source:raise RuntimeError('compact replay differs')
            results.append(dict(tokens=count,observations=2*count,legacy_uncompressed_bytes=old_bytes,
                                compact_uncompressed_bytes=logical_bytes,gzip_bytes=path.stat().st_size,
                                exact_replay=True))
    report=dict(scope='synthetic cumulative SSE; 6 characters/token, native and client observations',
                results=results,compressed_ratio_is_not_the_linear_growth_test=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
