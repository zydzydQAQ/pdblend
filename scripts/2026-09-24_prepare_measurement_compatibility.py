#!/usr/bin/env python3
"""Review two frozen source manifests for the narrowly allowed frequency repair."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pdblend.bench.measurement_compatibility import prepare_compatibility
from pdblend.bench.resident_session import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-source', type=Path, required=True)
    parser.add_argument('--right-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review = prepare_compatibility(args.left_source, args.right_source)
    write_new(args.out, review)
    print(json.dumps(dict(path=str(args.out.resolve()), compatible=review['compatible'],
                         changed_files=review['changed_files'])))


if __name__ == '__main__':
    main()
