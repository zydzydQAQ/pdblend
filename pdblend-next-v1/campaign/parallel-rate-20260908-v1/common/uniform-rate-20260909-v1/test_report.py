"""CPU regressions for evidence fidelity, missing values, and plot boundaries."""
import csv
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import contract as c
import monitor
import report

class ReportTests(unittest.TestCase):
    def test_pending_second_repeat_caps_chart_at_first_loss(self):
        decision = {'phase': 'pdblend', 'rate_rps': 3, 'decision': {'cap_observed': True}}
        self.assertEqual(report.cap_rate(decision), 3)
        self.assertIsNone(report.cap_rate({'phase': 'pdblend', 'rate_rps': 3, 'decision': {'cap_observed': False}}))
        self.assertEqual(report.relevant_positions({'positions': [{'rate_rps': 3}, {'rate_rps': 6}]}, decision), [{'rate_rps': 3}])

    def test_stop_star_marks_first_failing_repeat_not_minimum(self):
        common=dict(model='7b',dataset='alpaca',system='pdblend',rate_rps=18,measurement_valid=True,work_complete=True)
        rows=[dict(common,repeat=2,slo_attainment=.7),dict(common,repeat=1,slo_attainment=.89)]
        self.assertEqual(report.stop_trigger(rows,'7b','alpaca',18)['repeat'],1)
        self.assertEqual(report.stop_trigger(rows,'7b','alpaca',18)['slo_attainment'],.89)
        rows[1]['slo_attainment']=.9
        self.assertEqual(report.stop_trigger(rows,'7b','alpaca',18)['repeat'],2)

    def test_export_keeps_missing_blank_and_marks_partial(self):
        result = dict(complete=False, metric_audit_errors=[], groups=[dict(model='7b', dataset='alpaca',
            rate_step_rps=3, rate_grid=[3,6], cap_rate_rps=3, complete=False,
            decision=dict(phase='pdblend', rate_rps=3, decision=dict(cap_observed=True)))],
            observations=[dict(model='7b', dataset='alpaca', system='distserve', rate_rps=3,
                repeat=1, work_complete=False, token_throughput_is_exact=False, recorded_output_is_partial=True,
                cell_id='one', slo_attainment=.5, energy_j=12, ttft_avg_s=None,
                measurement_duration_s=100, energy_per_gpu_j=[1]*8, gpu_util_per_gpu=[.5]*8,
                checkpoint=dict(path='/root/workspace/cp.json',sha256='a'*64),
                raw_power=dict(path='/root/workspace/power.csv',sha256='b'*64)),
                dict(model='7b', dataset='alpaca', system='pdblend', rate_rps=6,
                     cell_id='two', work_complete=True, slo_attainment=.99)])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory); rows=report.export(result,out)
            self.assertEqual(len(rows),1)
            self.assertIsNone(rows[0]['ttft_avg_s_mean'])
            self.assertEqual(rows[0]['incomplete_work_repeats'],1)
            self.assertEqual(rows[0]['partial_token_repeats'],1)
            with (out/'measurements.csv').open() as stream: saved=list(csv.DictReader(stream))
            self.assertEqual(saved[0]['ttft_avg_s'],'')
            self.assertEqual(saved[0]['recorded_output_is_partial'],'True')
            with (out/'raw-evidence-index.csv').open() as stream: evidence=list(csv.DictReader(stream))
            self.assertEqual(evidence[0]['checkpoint_sha256'],'a'*64)
            self.assertEqual(evidence[0]['raw_power_sha256'],'b'*64)
            self.assertEqual(evidence[1]['in_current_grid_report'],'False')
            with (out/'gpu-details.csv').open() as stream: gpus=list(csv.DictReader(stream))
            self.assertEqual(len(gpus),16)
            self.assertEqual([r['gpu_index'] for r in gpus[:8]],list(map(str,range(8))))
            self.assertEqual(gpus[0]['energy_j'],'1')
            self.assertEqual(gpus[0]['gpu_util_percent'],'50.0')
            self.assertEqual(gpus[0]['average_power_w'],'0.01')
            self.assertEqual(gpus[8]['energy_j'],'')

    def test_raw_cache_rechecks_changed_evidence_and_strict_slo(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cell=root/'results/cells/cell'; receipt=root/'results/operations/cell/receipt.json'
            cell.mkdir(parents=True);receipt.parent.mkdir(parents=True);receipt.write_text('{}')
            trace=root/'trace.json';trace.write_text('{}')
            for name in ('bench.csv','power.csv','summary.json'): (cell/name).write_text('{}')
            artifacts={str(p):c.sha(p) for p in cell.iterdir()};checkpoint=root/'checkpoint.json'
            checkpoint.write_text(json.dumps(dict(receipt=c.ref(receipt),row=dict(cell_id='cell',trace=str(trace)),artifacts=artifacts)))
            observation=dict(checkpoint=c.ref(checkpoint),slo_attainment=.9,work_complete=True)
            raw=dict(slo_attainment=.9,work_complete=True,request_throughput_rps=1,token_throughput_tps=2,generated_token_count_complete=True)
            with patch.object(report.m,'audit_receipt',return_value=raw) as audit:
                self.assertEqual(report.audit_observation(observation,root/'cache')['slo_attainment'],.9)
                report.audit_observation(observation,root/'cache');self.assertEqual(audit.call_count,1)
                with self.assertRaisesRegex(ValueError,'strict raw'): report.audit_observation(dict(observation,slo_attainment=1),root/'cache')
                (cell/'bench.csv').write_text('changed')
                with self.assertRaisesRegex(ValueError,'artifact changed'): report.audit_observation(observation,root/'cache')

    def test_mirror_never_overwrites_conflicting_immutable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(monitor,'WORKSPACE',root):
                target=root/'evidence.json';target.write_text('existing');incoming=root/'incoming';incoming.write_text('new')
                mirror=monitor.Mirror(root/'cache.json');ref=dict(path=str(target),sha256=c.sha(incoming))
                with self.assertRaisesRegex(ValueError,'conflict'): mirror.install(ref,incoming)
                self.assertEqual(target.read_text(),'existing');target.unlink();mirror.install(ref,incoming)
                self.assertEqual(target.read_text(),'new');self.assertTrue(mirror.present(ref))

    def test_completed_child_repeat_is_visible_before_stage_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); pipeline=root/'pipeline-001/status.json'
            child=root/'measurements/r3/status.json'; child.parent.mkdir(parents=True)
            ref=dict(path='/root/workspace/audit.json',sha256='a'*64)
            child.write_text(json.dumps(dict(complete=False, observations=[ref])))
            state=dict(observations=[],stages=dict(measure=dict(argv=['python3','/x/run_cells.py','--out',str(child.parent)])))
            self.assertEqual(report.pipeline_observation_refs(pipeline,state),[ref])
            state['observations']=[ref]
            self.assertEqual(report.pipeline_observation_refs(pipeline,state),[ref])

    def test_legacy_setup_index_only_pins_terminal_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); status=root/'uniform-test/restore/status.json'
            power=status.parent/'power';power.mkdir(parents=True)
            for name in ('power.csv','power_source.json','power_metadata.jsonl'):
                (power/name).write_text('evidence')
            value=dict(complete=False,setup_and_correctness_energy_j=5,measurement_end_s=2)
            status.write_text(json.dumps(value))
            def index():
                result=subprocess.run([sys.executable,'-c',monitor.INDEX_SCRIPT,str(root)],capture_output=True,check=True)
                return json.loads(result.stdout)[0]
            self.assertEqual(index()['extra_refs'],[])
            value.update(complete=True,finished_s=3);status.write_text(json.dumps(value))
            refs=index()['extra_refs'];self.assertEqual(len(refs),3)
            for ref in refs:self.assertEqual(ref['sha256'],c.sha(ref['path']))

    def test_pdb_scope_requires_all_raw_boundaries_terminal_supervisors_and_idle_mirror(self):
        result=dict(complete=False,groups=[dict(pdb_boundary_complete=True) for _ in range(9)])
        states={model:(Path('/tmp')/model/'pipeline/status.json',dict(complete=True,finished_s=3,node_lease_held=False))
                for model in c.MODELS}
        value=monitor.completion_state(result,'pdblend',states=states,guards=[])
        self.assertTrue(value['scope_complete']);self.assertFalse(value['five_system_complete'])
        self.assertFalse(monitor.completion_state(result,'five_systems',states=states,guards=[])['scope_complete'])
        self.assertFalse(monitor.completion_state(result,'pdblend',states=states,guards=[],hydration_active=True)['scope_complete'])
        states['7b'][1].update(complete=False,error='stop requested at stage boundary')
        self.assertFalse(monitor.completion_state(result,'pdblend',states=states,guards=[])['scope_complete'])
        guard=dict(pipeline=str(states['7b'][0].parent),complete=True,pdb_complete=True,finished_s=4)
        self.assertTrue(monitor.completion_state(result,'pdblend',states=states,guards=[guard])['scope_complete'])
        result['groups'][0]['pdb_boundary_complete']=False
        self.assertFalse(monitor.completion_state(result,'pdblend',states=states,guards=[guard])['scope_complete'])

    def test_failed_attempt_remains_separate_after_successor_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old=root/'A/uniform-test/pipeline-001/status.json';old.parent.mkdir(parents=True)
            old.write_text(json.dumps(dict(finished_s=2,error='native clock unavailable',observed_checkpoints=[])))
            new=root/'A/uniform-test/pipeline-002/status.json';new.parent.mkdir(parents=True)
            new.write_text(json.dumps(dict(complete=False,phase='pdblend',observations=[])))
            result=report.collect_engineering_failures(root/'cache',root)
            self.assertEqual(len(result['entries']),1)
            failure=result['entries'][0]
            self.assertEqual(failure['classification'],'needs_diagnosis')
            self.assertFalse(failure['included_in_main_curves']);self.assertFalse(failure['used_as_capacity_boundary'])
            self.assertEqual(failure['status']['sha256'],c.sha(old))
            report.export_engineering_failures(result,root)
            with (root/'engineering-failures.csv').open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['status_path'],str(old))
            self.assertEqual(rows[0]['checkpoint_path'],'')

    def test_diagnosis_binds_checkpoint_not_only_retry_cell_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';raw.write_text('proof')
            cp=dict(path=str(root/'old-checkpoint'),sha256='b'*64)
            path=root/'diagnosis.json';path.write_text(json.dumps(dict(cell_id='retry-same-id',
                classification='engineering_configuration_idle_frequency_domain',measurement=dict(checkpoint=cp),
                evidence=[c.ref(raw)],slo_boundary_eligible=False,valid_performance_repeat=False)))
            refs=[c.ref(path)];diagnoses,errors=report.checked_diagnoses(root/'cache',refs)
            self.assertFalse(errors);self.assertIn(('retry-same-id',cp['path'],cp['sha256']),diagnoses)
            self.assertNotIn(('retry-same-id','new-checkpoint','c'*64),diagnoses)
            raw.write_text('tampered')
            diagnoses,errors=report.checked_diagnoses(root/'cache',refs)
            self.assertFalse(diagnoses);self.assertIn('diagnosis evidence changed',errors[0]['error'])

    def test_traversal_ignores_mutable_source_and_unpinned_logs(self):
        digest='a'*64
        value=dict(observations=[dict(path='/root/workspace/audit.json',sha256=digest)],source_files={'/root/workspace/runtime.py':digest},
            log='/root/workspace/active.log',qualification_validator=dict(path='/root/workspace/verify.py',sha256=digest),
            artifacts={'/root/workspace/final.log':digest})
        self.assertEqual({r['path'] for r in monitor.references(value)}, {'/root/workspace/audit.json','/root/workspace/final.log'})
        self.assertEqual(list(monitor.references(dict(positions=[],cells=[],reused_observations=[value]))),[])

if __name__=='__main__': unittest.main()
