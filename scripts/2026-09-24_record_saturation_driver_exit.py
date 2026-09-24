#!/usr/bin/env python3
"""Record systemd's exit diagnosis without collecting unrelated environment."""
import json
import os
from pathlib import Path
import sys
import time

out = Path(sys.argv[1]) / 'orchestration'
row = dict(at_s=time.time(), service_result=os.environ.get('SERVICE_RESULT'),
           exit_code=os.environ.get('EXIT_CODE'), exit_status=os.environ.get('EXIT_STATUS'))
with (out / 'driver-exits.jsonl').open('a') as stream:
    stream.write(json.dumps(row, sort_keys=True) + '\n')
