"""Read-only full-profile failure audit; retains complete work and clock negative."""
import bisect,collections,csv,hashlib,importlib.util,json,pathlib,socket,time
R=pathlib.Path(__file__).resolve().parent.parent
D=R/"B/distributed-14b-v1/frequency2400-profile-002"
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
s=read(D/"status.json");assert s["finished_s"] is not None and s["errors"]==["ValueError('loaded SM outside original +/-15 MHz command band')"]
p=s["current_point"];rawpath=D/p["point_id"]/"raw.json";raw=read(rawpath);events=[json.loads(x) for x in pathlib.Path(raw["events"]["path"]).read_text().splitlines()]
starts=[e["started_s"] for e in events]
with (D/"power/clocks.csv").open() as f:clocks=[(float(r["t_s"]),[float(r[f"gpu{i}_sm_mhz"]) for i in range(8)]) for r in csv.DictReader(f)]
active=[];bad=[]
for t,values in clocks:
 k=bisect.bisect_right(starts,t)-1
 if k>=0 and events[k]["request_ids"] and t<=events[k]["finished_s"]:
  value=values[p["gpu"]];active.append(value)
  if abs(value-p["frequency_mhz"])>15:bad.append(dict(t_s=t,actual_mhz=value,event_index=k,prefill=events[k]["prefill"],decode=events[k]["decode"],tokens=events[k]["tokens"]))
assert active and bad
v=load(R/"common/distributed14b-profile-validation-v1/validate.py","full2400native")
# Native validator still rejects the actual frequency; output/cleanup are separately reconstructed.
work=all(r.get("success") is True and r.get("done_marker") is True and r.get("http_status")==200 and len(r["output_token_ids"])==len(r["token_received_s"])==p["output_tokens"] and r["usage"]["completion_tokens"]==p["output_tokens"] for r in raw["requests"])
assert work and len(raw["requests"])==p["batch"]
v.cleanup_saved(raw["cleanup"],p["instance_id"])
identity=read(D/"identity.after.json")
for item in identity:v.native_saved(item["runtime"],item["provenance"]["instance_id"],8192)
q=load(R/"common/distributed14b-qualification-v1/verify.py","full2400rawpower")
energy,_=q.power_operation(D,s,read(D/"spec.json")["host_manifest"])
assert s["clock_restore_complete"] is True and all(x["complete"] and not x["errors"] for x in s["native_cleanup"])
out=R/"B/distributed-14b-v1/full2400-frequency-diagnosis-001.json";assert not out.exists()
x=dict(schema="distributed14b-full2400-frequency-negative-v1",classification="complete_native_work_failed_loaded_clock_qualification",hostname=socket.gethostname(),read_s=time.time(),GPU_actions=False,
 status=ref(D/"status.json"),raw=ref(rawpath),native_events=raw["events"],clock_stream=ref(D/"power/clocks.csv"),power_source=ref(D/"power/power_source.json"),profile_driver=ref(R/"common/distributed14b-frequency-profile-v2/manifest.json"),source=ref(pathlib.Path(__file__).resolve()),
 actual_point=p,preceding_valid_observations=len(s["points"]),full_group_complete=False,profile_published=False,native_work_complete=work,request_count=len(raw["requests"]),actual_completed_output_tokens=sum(len(r["output_token_ids"]) for r in raw["requests"]),
 native_cleanup_complete=True,clock_restore_complete=True,raw_energy_measurement_valid=True,whole_operation_raw_energy=energy,original_measurement_valid=s["measurement_valid"],
 actual_loaded_clock=dict(samples=len(active),min_mhz=min(active),max_mhz=max(active),outside_original_15mhz_band=len(bad),counts=dict(collections.Counter(active)),bad_samples=bad),scientific_comparison_eligible=False,not_a_performance_rate_point=True,automatic_retry=False)
out.write_text(json.dumps(x,indent=2)+"\n");print(json.dumps(dict(output=ref(out),valid_prefix=len(s["points"]),work_complete=work,clock=x["actual_loaded_clock"]|dict(bad_samples=bad[:3]))))
