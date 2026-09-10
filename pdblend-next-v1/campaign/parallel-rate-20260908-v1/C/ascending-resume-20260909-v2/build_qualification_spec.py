"""Freeze C TP1 qualification source separately from the reviewed cold package."""
import json
from pathlib import Path
import time
import cold_restore as c


def main():
    path=c.HERE/'qualification-spec.json'
    assert not path.exists()
    coldref=c.ref(c.HERE/'cold-spec.json');cold=c.checked(coldref)
    c.validate(cold)
    host=Path(cold['host_manifest']['path']).parent
    refs=dict(cold_spec=coldref,capacity_executor=c.ref(host/'capacity_executor.py'),
              capacity_backend=c.ref(host/'capacity_backend.py'),
              stream=c.ref(c.ROOT.parents[1]/'campaign/main-slo-improvement-v1/A/long-batch6-code-004/stream.py'))
    files=dict(cold['files'])
    for reference in refs.values():files[reference['path']]=reference['sha256']
    for name in ('qualify.py','verify_qualification.py','build_qualification_spec.py'):
        files[str(c.HERE/name)]=c.sha(c.HERE/name)
    spec=dict(schema='C7B-frequency-legacy-cancel-spec-v1',hostname=c.HOSTNAME,node='C',created_s=time.time(),
              frequencies_mhz=[900,1500,2100,2520],input_tokens=128,output_tokens=64,
              controlled_cancel_output_tokens=512,fresh_restoration_reference_required=True,
              gpu_work_started=False,**refs,files=files)
    c.save(path,spec)
    print(json.dumps(dict(spec=c.ref(path),files=len(files),gpu_work_started=False)))


if __name__=='__main__':main()
