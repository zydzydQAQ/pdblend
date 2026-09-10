"""Attempt a CPU proof cache; preserve the original full verifier on build failure."""
import argparse,hashlib,json,os,signal,subprocess,sys
from pathlib import Path

def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def execute(qualification,validator,helper,out):
    out=Path(out);assert not out.exists(),'fresh optional cache attempt required'
    out.mkdir(parents=True)
    interrupted=[]
    with (out/'build.stdout').open('xb') as stdout,(out/'build.stderr').open('xb') as stderr:
        proc=subprocess.Popen([sys.executable,'-B',str(helper),'--qualification',str(qualification),
            '--validator',str(validator),'--out',str(out/'cache')],stdout=stdout,stderr=stderr)
        def stop(sig,frame):
            interrupted.append(sig)
            if proc.poll() is None:proc.send_signal(sig)
        previous={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:code=proc.wait()
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
    result=dict(schema='optional-qualification-cache-selection-v1',qualification=ref(qualification),
        original_validator=ref(validator),helper=ref(helper),exitcode=code,interrupted=bool(interrupted),
        build_stdout=ref(out/'build.stdout'),build_stderr=ref(out/'build.stderr'))
    if not interrupted and code==0:
        built=json.loads((out/'build.stdout').read_text())
        assert built['qualification']==result['qualification'] and built['helper']==result['helper']
        assert built['qualification_validator']==ref(out/'cache/verify.py') and built['cache']==ref(out/'cache/cache.json')
        result.update(cache_created=True,qualification_validator=built['qualification_validator'],cache=built['cache'])
    elif not interrupted and code>0:
        # This makes no qualification claim. The original full verifier must
        # still pass inside meter_binding before any formal handoff is issued.
        result.update(cache_created=False,qualification_validator=result['original_validator'],
            fallback_requires_original_full_verifier=True,
            reason='Cache was not built; preserve diagnostics and use unchanged full independent audit.')
    with (out/'selection.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    if interrupted or code<0:raise SystemExit(128+(interrupted[-1] if interrupted else -code))
    return result

def main():
    ap=argparse.ArgumentParser()
    for name in ('qualification','validator','helper','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();print(json.dumps(execute(a.qualification,a.validator,a.helper,a.out)))
if __name__=='__main__':main()
