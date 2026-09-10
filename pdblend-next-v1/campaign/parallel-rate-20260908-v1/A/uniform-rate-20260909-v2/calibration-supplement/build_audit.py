from pathlib import Path
H=Path(__file__).resolve().parent;old=H.parents[1]/'uniform-rate-20260909-v1/dynamic-producer/audit.py'
s=old.read_text()
s=s.replace('import fresh_support as f',"import sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[2]/'uniform-rate-20260909-v1/dynamic-producer'))\nimport fresh_support as f",1)
s=s.replace('def audit_stage(out, spec_ref, mode):','def audit_stage(out, spec_ref, mode, *, supplement=False, exclude_gap=False):',1)
s=s.replace("    expected = 27 if mode == 'layout_calibration' else 1", "    f.need(mode=='layout_calibration', 'mixed calibration terminal audit only')\n    expected = 8 if supplement else 27\n    f.need(not (supplement and exclude_gap), 'new supplement cannot exclude invalid raw')\n    excluded=[]",1)
s=s.replace("            idle_result(reference, cap['identity'], inventory)","""            if exclude_gap and Path(reference['path']).parent.name=='cycle-3-idle-layout2':
                try:
                    idle_result(reference,cap['identity'],inventory)
                except ValueError as exc:
                    f.need(str(exc)=='native samples do not continuously cover the exact declared idle window','unexpected old idle failure')
                    excluded.append(dict(result=reference,reason=str(exc),used_as_qualification_evidence=False))
                else:
                    raise ValueError('expected preserved idle gap negative not reproduced')
            else:
                idle_result(reference, cap['identity'], inventory)""",1)
s=s.replace("    return dict(schema='new-A-fresh-capacity-stage-audit-v1', passed=True, independently_recomputed=True,", "    f.need(not exclude_gap or len(excluded)==1,'one preserved idle-gap exclusion required')\n    if supplement:\n        f.need([Path(r['path']).parent.name for r in state['completed']]==spec['selected_phase_names'],'exact eight supplement windows required')\n    return dict(schema='new-A-mixed-calibration-terminal-audit-v2', passed=True, independently_recomputed=True,\n                supplement=supplement, excluded_measurements=excluded, all_phases_qualified=False,",1)
with (H/'audit_support.py').open('x') as stream:stream.write(s)
