"""Validate this CPU package and its exact existing dependencies; no live calls."""
import json
from pathlib import Path
from gate_evidence import read,require,sha

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    manifest=read(ROOT/'manifest.json')
    for path,h in manifest['files'].items():require(sha(ROOT/path)==h,'package changed: '+path)
    for path,h in read(ROOT/'source-contract.json')['files'].items():require(sha(path)==h,'dependency changed: '+path)
    print(json.dumps(dict(cpu_only=True,network_actions=False,hardware_actions=False,manifest_sha256=sha(ROOT/'manifest.json'),files=len(manifest['files']))))
