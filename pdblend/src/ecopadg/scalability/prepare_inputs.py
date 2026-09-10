"""Export explicit corpus splits; record eligibility exclusions before runs."""
import argparse
from pathlib import Path

from .artifacts import object_hash, read_json, sha256, write_json
from .workload import _normalize_record


def export_pool(source, dataset, out, *, split='formal'):
    payload = read_json(source)
    if payload.get('dataset') != dataset:
        raise ValueError('corpus dataset does not match the declaration')
    records = payload[split]
    if isinstance(records, dict):
        records = [r for seed in sorted(records) for r in records[seed]]
    seen = set()
    selected, exclusions = [], []
    duplicates = 0
    for index, record in enumerate(records):
        identity = object_hash(record)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        if record.get('output_len',record.get('output_tokens')) == 1:
            exclusions.append(dict(source_index=index, record_sha256=identity,
                                   reason='output length 1 has no after-first-token TPOT interval'))
            continue
        _normalize_record(record)  # Other malformed records are not silently dropped.
        selected.append(record)
    if not selected:
        raise ValueError('no eligible requests remain')
    result = dict(dataset=dataset, records=selected, exclusions=exclusions,
        source_path=str(Path(source).resolve()), source_sha256=sha256(source), source_split=split,
        deduplicated_identical_records=duplicates, source_record_count=len(records),
        eligibility='output_tokens >= 2, fixed before any performance measurement',
        sampling='uniform with replacement with independently seeded arrival and content sampling')
    write_json(out,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset',choices=['sharegpt','longbench'],required=True)
    parser.add_argument('--split',default='formal')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=export_pool(args.source,args.dataset,args.out,split=args.split)
    print(f"{len(result['records'])} eligible; {len(result['exclusions'])} declared exclusions")


if __name__=='__main__':
    main()
