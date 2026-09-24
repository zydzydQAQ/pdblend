#!/usr/bin/env python3
"""Import and validate a frozen recovery group inside its runtime image, without GPUs."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from pdblend.bench.comparison_runtime import pdblend_window_resources

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--group', type=Path, required=True)
args = parser.parse_args()
group = json.loads(args.group.read_text())
specs = [SimpleNamespace(tp=r['tp'], pp=r['pp'], generation=0)
         for r in group['engine_identity']['instances']]
for point in group['points']:
    loaded, plan = pdblend_window_resources(point, specs)
    print(json.dumps(dict(point=point['name'], loaded=True, tp=plan.tp,
                         profile=loaded.profile_key, hardware_executed=False)))
