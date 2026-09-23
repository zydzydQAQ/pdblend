"""Versioned revalidation of existing pinned author CSV bytes, never a download claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from .history_adapter import PINNED_ASSETS, prepare_trace, save, sha

REPOSITORY = 'Azure/AzurePublicDataset'
REVISION = '207bed67dd10090b28ad4f745b2cfd41a11aace4'
REFERENCES = {
    'AzureLLMInferenceDataset2024.md': 'b4018009abc757d3ec77300253c4dc93dfe85709bc3455998e3d6d4c12004c8e',
    'LICENSE': '9bbc1ea9fe5c96df01b311a2ac864d5b18fc87b9948bfd14770e4b44db755ee9',
    'LICENSE-CODE': '91f7258203fd43ba6d6f7f72b644eaedee9e3430f035c053beb6f34052f045e2',
    'README.md': '4b21c761d1f6d5f5e8c64af65e57a02232669e63512a63f63a9a5640d92bbde9',
}


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'dynamollm-source-revalidation/1'}), timeout=60) as response:
        return response.read()


def prepare(source, out, *, previous_history=None, get=fetch):
    """Re-read raw CSV and regenerate counts with the current pinned adapter.

    An absent old download receipt remains absent. The new receipt explicitly
    documents existing-file verification at this time, including freshly read
    pinned reference documents. No GPU results inherit qualification from it.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    expected = PINNED_ASSETS.get(source.name)
    if expected is None or not source.is_file() or sha(source) != expected:
        raise ValueError('pinned existing author source SHA differs')
    out.mkdir(parents=True, exist_ok=False)
    references = {}
    for name, digest in REFERENCES.items():
        url = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}'
        path = out/name
        path.write_bytes(get(url))
        if sha(path) != digest:
            raise ValueError('pinned author reference SHA differs: '+name)
        references[str(path)] = digest
    receipt_path = out/'source-revalidation.json'
    save(receipt_path, dict(schema='dynamo-existing-source-revalidation-v1',
        operation='verify_existing_file', downloaded=False, prior_download_receipt_restored=False,
        verified=True, verified_at_s=time.time(), path=str(source), sha256=expected,
        bytes=source.stat().st_size, repository=REPOSITORY, revision=REVISION,
        official_reference_files=references,
        previous_history=(dict(path=str(Path(previous_history).resolve()), sha256=sha(previous_history))
                          if previous_history else None),
        formal_eligible=False))
    provenance = dict(repository=REPOSITORY, revision=REVISION,
        source_receipt=str(receipt_path), source_receipt_sha256=sha(receipt_path),
        source_receipt_kind='existing_file_revalidation', official_reference_files=references,
        service='code' if '_code_' in source.name else 'conv',
        attribution='DynamoLLM author Azure arrival dataset; CC-BY')
    summary = prepare_trace(source, out/'history', expected_sha256=expected, provenance=provenance)
    from .validation import verified_history
    _, _, checked = verified_history(out/'history/history.json')
    result = dict(schema='dynamo-history-closure-v2', ready=True, source_receipt=str(receipt_path),
        history=str(out/'history/history.json'), history_sha256=sha(out/'history/history.json'),
        raw_reaggregated=True, previous_history_unchanged=True,
        counts=summary['rows'], chronological_holdout_available=checked['chronological_holdout_available'],
        calibration_history_usable=True, prediction_accuracy_qualified=False,
        formal_eligible=False, evidence=checked['evidence'])
    save(out/'completion.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous-history', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.out, previous_history=args.previous_history)))


if __name__ == '__main__':
    main()
