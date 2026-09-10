"""Copy only already-recorded engine events; no hardware/service calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[2] / 'B32B-five-system100-baseline-deployment-v1/runtime'
START, END = 1788929935.4457362, 1788930148.5381398

if __name__ == '__main__':
    out = HERE / 'repeat2-runtime-window'
    out.mkdir(exist_ok=True)
    index = []
    for source in sorted(RUNTIME.glob('*.control.events.jsonl')):
        digest = hashlib.sha256()
        target = out / source.name
        count = 0
        with source.open('rb') as stream, target.open('xb') as dest:
            for raw in stream:
                digest.update(raw)
                if not raw.endswith(b'\n'):
                    continue
                row = json.loads(raw)
                if row.get('finished_s', 0) >= START and row.get('started_s', END + 1) <= END:
                    dest.write(raw)
                    count += 1
        index.append(dict(source=str(source), source_sha256_at_capture=digest.hexdigest(),
            window_path=str(target), window_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), records=count))
    assert len(index) == 4 and all(x['records'] for x in index)
    (HERE / 'repeat2-runtime-window-index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index, indent=2))
