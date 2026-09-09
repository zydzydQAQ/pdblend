"""Update only the actual-host experiment index; all previous versions are kept."""
import json,hashlib,socket,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def update(binding_path,manifest_path,pid,phase,datasets,supervisor_status,exitcode=None):
    b=read(binding_path);m=read(manifest_path)
    if b['hostname']!=socket.gethostname():raise RuntimeError('index must be written on the actual bound host')
    if exitcode is None:
        cmd=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
        if str(binding_path).encode() not in cmd or str(manifest_path).encode() not in cmd:raise RuntimeError('actual queue command does not bind these inputs')
    root=Path('/root/workspace/pdblend-next-v1/campaign');current=root/'current-experiment.json';old=read(current)
    history=root/'current-experiment-history'/(str(time.time_ns())+'-baseline100-'+phase);history.mkdir(parents=True)
    for name in ('current-experiment.json','CURRENT_EXPERIMENT.md'):
        p=root/name
        if p.exists():(history/name).write_bytes(p.read_bytes())
    value=dict(old);value.update(schema=3,written_s=time.time(),hostname=b['hostname'],model=b['model'],protocol_id=b['protocol_id'],active_system=b['system'],implementation_variant=b['implementation_variant'],active_phase=phase,selected_datasets=datasets,active_system_main_cells=10*len(datasets),active_system_scale_cells=6*len(datasets),queue_pid=pid,queue_running=exitcode is None,queue_exitcode=exitcode,supervisor_status=str(supervisor_status),binding=dict(path=str(binding_path),sha256=sha(binding_path)),workload_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),host_release=b['host_release'],actual_controller_configs=b['configs'],historical_index=str(history),correctness_evidence=b.get('correctness_evidence'),output_correctness_verified=b.get('output_correctness_verified'),baseline_policy='paired 100s baseline measurements; all historical and failed points retained')
    value.pop('bridge',None)
    tmp=root/'current-experiment.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(current)
    (root/'CURRENT_EXPERIMENT.md').write_text('# Current actual-host experiment\n\n'+f'Model: {b["model"]}. System: {b["implementation_variant"]}. Phase: {phase}. Datasets: {", ".join(datasets)}. Queue running: {exitcode is None}.\n\n'+'Seed 701; 100-second arrivals; request timeout and post-arrival drain cap 120 seconds. All eight GPUs measured, including idle and failed work.\n\n'+f'Actual controller configurations: {b["configs"]}\n\nBinding: {binding_path}\n\nSupervisor: {supervisor_status}\n\nOriginal deadline: 2026-09-08 21:06:10 CST.\n')
    return dict(path=str(current),sha256=sha(current),queue_running=exitcode is None)
