#!/usr/bin/env python3
"""Audit preregistered 900 MHz holdout workers, provenance and stable evidence."""
import argparse
import json
from pathlib import Path
from pdblend.profile.model import PerfModel
from pdblend.profile.validation import audit_independent

p=argparse.ArgumentParser()
p.add_argument('candidate', type=Path)
p.add_argument('validation', type=Path)
p.add_argument('out', type=Path)
a=p.parse_args()
result=audit_independent(a.candidate,a.validation,PerfModel.load(a.candidate))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=1))
print(json.dumps(dict(gate=result['gate'],failures=result['failures'],errors=result['errors']),indent=1))
raise SystemExit(0 if result['gate']['status']=='PASS' else 2)
