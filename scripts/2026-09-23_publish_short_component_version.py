#!/usr/bin/env python3
"""Publish an experimental short mixed component only after raw holdout audits."""
import argparse
import json
from pathlib import Path
from pdblend.profile.short_version import publish

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',type=Path,required=True)
p.add_argument('--archive',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();v=publish(**vars(a))
print(json.dumps(dict(version_id=v['version_id'],version=str(a.out/'version.json'),
    formal_eligible=False,usage='experimental',holdout_maximum_error=v['audit']['maximum_error']),indent=2))
