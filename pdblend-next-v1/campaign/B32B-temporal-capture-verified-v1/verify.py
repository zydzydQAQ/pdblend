"""CPU-only postprocess of terminal capture; preserve original wrapper import error."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent;P=C/'B32B-temporal-observation-execution-v3'
sys.path.insert(0,str(P));import common as c
c.require(c.sha(P/'manifest.json')=='8b25efcfc61952691af405ab93693856ba709f9b873c9098c58180fd960a696d','frozen execution source differs');c.package_check()
proof=c.read(ROOT/'actual-terminal.json');c.require(proof['hostname']==c.NODE and not any(proof['processes_live'].values()),'actual B HTTP/diagnostic processes not terminal');c.verify_files(proof['files']);before=dict(proof['files'])
p=C/'B32B-temporal-observation-attempt-002/results';parent=c.read(p/'status.json');child=c.read(p/'child/status.json');c.require(parent['complete'] and parent['all_original_restored'] and parent['measurement_valid'],'actual operation/native restoration incomplete');c.require(parent['capture_error']=='''ModuleNotFoundError("No module named 'pdblend_diagnostics'")''','only separately repaired offline import error')
# Load exactly the original frozen helper under the name its original verifier imports.
spec=importlib.util.spec_from_file_location('pdblend_diagnostics',c.CANDIDATE/'pdblend_diagnostics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['pdblend_diagnostics']=module
result=c.frozen_capture(p.parent/'observation-spec.json',p/'capture-live',ROOT/'capture-frozen',p/'child/full-outputs.json',child,parent['process_terminal'])
c.verify_files(before)
report=dict(schema=1,checked_s=time.time(),capture_verified=result['capture_complete'],original_wrapper_observation_completed=parent['observation_completed'],original_wrapper_error_preserved=parent['capture_error'],same_helper_sha256=c.sha(c.CANDIDATE/'pdblend_diagnostics.py'),same_verifier_sha256=c.sha(c.CANDIDATE/'verify_capture.py'),hardware_actions=False,performance_evidence=False,hardware_correctness_proven=False,original_exact_gate_passed=child['exact_passed'],operation_measurement_valid=parent['measurement_valid'],result=result,input_files=before,actual_terminal_sha256=c.sha(ROOT/'actual-terminal.json'))
c.write(ROOT/'report.json',report);print(json.dumps({k:report[k] for k in ('capture_verified','original_wrapper_observation_completed','original_exact_gate_passed','hardware_correctness_proven')}))
