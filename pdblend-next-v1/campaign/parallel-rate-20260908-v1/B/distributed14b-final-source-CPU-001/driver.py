import subprocess,json,pathlib,os,time,hashlib
r=pathlib.Path("/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1");out=r/"B/distributed14b-final-source-CPU-001";results=[]
for name,script in (("qualification19","common/distributed14b-qualification-v1/test_verify.py"),("legacy20","common/legacy55-at-qualified-max-v1/test_max_gate.py"),("registration26","common/distributed14b-frequency-registration-v1/test_registration.py")):
 p=subprocess.run(["python3","-B",str(r/script)],capture_output=True,text=True,timeout=120)
 (out/(name+".log")).write_text(p.stdout+p.stderr);results.append(dict(name=name,exitcode=p.returncode,script=script,sha256=hashlib.sha256((r/script).read_bytes()).hexdigest()))
 (out/"status.json").write_text(json.dumps(dict(pid=os.getpid(),hostname=os.uname().nodename,results=results,complete=len(results)==3,passed=all(x["exitcode"]==0 for x in results),finished_s=time.time() if len(results)==3 else None),indent=2)+"\n")
