import hashlib,json,os,pathlib,socket,subprocess,sys,time
B=pathlib.Path(__file__).resolve().parent;R=B.parent
inputs=B/'A-final43-C-saved15-inputs-v1.json';data=json.loads(inputs.read_text())
result=dict(schema='B-independent-A-final43-C-saved15-review-v1',physical_host=socket.gethostname(),pid=os.getpid(),started_s=time.time(),input_reference=dict(path=str(inputs),sha256=hashlib.sha256(inputs.read_bytes()).hexdigest()),GPU_actions=False,tests=[])
for p,h in data['files'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,'input bytes differ '+p
for relative,tests in [('A/final-p8-code-002',['test_runner.py','test_contract.py']),('C',['test_isolated_saved_audit_v1.py'])]:
 command=[sys.executable,'-B','-m','pytest',*tests,'-q','-p','no:cacheprovider'];started=time.time();done=subprocess.run(command,cwd=R/relative,text=True,capture_output=True)
 row=dict(directory=str(R/relative),argv=command,exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr,elapsed_s=time.time()-started);result['tests'].append(row);print(json.dumps(row),flush=True)
result.update(passed=all(x['exit_code']==0 for x in result['tests']),finished_s=time.time())
p=B/'independent-A-final43-C-saved15-review-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())),flush=True)
