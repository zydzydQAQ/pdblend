"""Install the frozen evidence-only client before controller/cell imports."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
MANIFEST=ROOT/'common/token-evidence-v2/manifest.json'
def install():
 import benchmarks.scripts
 from benchmarks.scripts import bench_vllm as previous
 manifest=json.loads(MANIFEST.read_text());ref=manifest['collector']
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 if sha(previous.__file__)!=manifest['predecessor']['sha256']:raise ValueError('collector predecessor differs from qualified native client')
 if sha(ref['path'])!=ref['sha256'] or not manifest['legacy_fields_unchanged'] or not manifest['control_source_unchanged']:raise ValueError('evidence collector manifest changed')
 name='benchmarks.scripts.bench_vllm';spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);benchmarks.scripts.bench_vllm=module
 return dict(schema='actual-formal-client-source-v2',collector=ref,manifest=dict(path=str(MANIFEST),sha256=sha(MANIFEST)),previous=dict(path=previous.__file__,sha256=sha(previous.__file__)),control_source_unchanged=True,legacy_metric_fields_unchanged=True)
