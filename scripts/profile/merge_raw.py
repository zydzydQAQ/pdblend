#!/usr/bin/env python3
"""Merge completed shards without rerunning GPU profiling."""
import argparse
import json
from pathlib import Path
from pdblend.profile.merge import merge_raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('raw', nargs='+', type=Path)
    args = ap.parse_args()
    raw = merge_raw(args.raw, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / 'raw.json'
    text = json.dumps(raw, indent=1)
    if target.exists() and target.read_text() != text:
        raise FileExistsError(f'refusing to overwrite different merged raw: {target}')
    target.write_text(text)
    print(json.dumps(dict(out=str(target), freqs=raw['freqs'],
                         **{s: len(raw[s]) for s in ('prefill', 'decode', 'mixed', 'transfer')})))


if __name__ == '__main__':
    main()
