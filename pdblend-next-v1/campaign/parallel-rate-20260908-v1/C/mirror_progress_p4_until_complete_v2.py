"""Passive remote C evidence mirror. Never issues GPU or experiment controls."""
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c_mirror_transport',HERE/'operate.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
outputs=[]
seen={}
while True:
    outputs=[HERE/'screen-p4',HERE/'p4-completion']+[HERE/'boundary-baselines-p4v2'/s for s in ('mixed','distserve','dynamollm','ecoserve')]+[HERE/p.name.removesuffix('-release') for p in sorted(HERE.glob('explore-p4-*-release'))]
    for out in outputs:
        try:
            code='import pathlib,json;f=pathlib.Path('+repr(str(out/'status.json'))+');print(f.read_text() if f.exists() else "null")'
            raw=m.remote(code);state=json.loads(raw)
            if state is None:continue
            key=(state.get('phase'),tuple(state.get('completed',[])),state.get('current_cell'))
            if seen.get(str(out))!=key and state.get('phase') != 'starting':
                m.sync(out)
                seen[str(out)]=key
                print(json.dumps(dict(captured_s=time.time(),out=str(out),phase=state.get('phase'),completed=len(state.get('completed',[])))),flush=True)
            out.mkdir(parents=True,exist_ok=True)
            tmp=out/'status.mirror-tmp';tmp.write_bytes(raw);tmp.replace(out/'status.json')
        except Exception as exc:
            print(json.dumps(dict(captured_s=time.time(),out=str(out),error=repr(exc))),flush=True)
    time.sleep(15)
