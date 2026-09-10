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
    def test_only_reviewed_same_host_rejection_contract_can_be_selected(self):
        self.assertIs(report.scheduling_contract({}, '14b', 'B'), c)
        state=dict(declaration_contract=dict(report.CAPACITY_CONTRACT))
        revised=report.scheduling_contract(state, '7b', 'C')
        observation=c.read(c.ROOT/'C/uniform-rate-20260909-v2/controller-rejection-reconstruction-001/observation.json')
        self.assertTrue(revised.acceptable_baseline(observation))
        self.assertFalse(c.acceptable_baseline(observation))
        with self.assertRaisesRegex(ValueError, 'physical scope'):
            report.scheduling_contract(state, '14b', 'Anew20260909')
        state['declaration_contract']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError, 'unreviewed scheduling contract'):
            report.scheduling_contract(state, '7b', 'C')

    def test_waiting_guard_does_not_hide_current_pipeline_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'B/uniform-rate-20260909-v2'
            current=base/'pipeline-006/status.json';guard=base/'q3-handoff-001/status.json'
            current.parent.mkdir(parents=True);guard.parent.mkdir(parents=True)
            common=dict(node='B',model='14b',datasets=['sharegpt'],scope='five_systems',declaration={})
            prior=dict(common,schema='uniform-v2-node-pipeline-status',observations=[{'path':'valid','sha256':'a'*64}])
            waiting=dict(common,schema='uniform-v2-B14-Q3-handoff-guard-status',observations=[])
            current.write_text(json.dumps(prior));guard.write_text(json.dumps(waiting))
            import os
            os.utime(current,ns=(1_000_000_000,1_000_000_000))
            os.utime(guard,ns=(2_000_000_000,2_000_000_000))
            key=('14b','sharegpt','B')
            with patch.object(report,'ROOT',root):
                self.assertEqual(report.latest_states()[key],(current,prior))
                adopted=dict(prior,phase='mixed')
                guard.write_text(json.dumps(adopted))
                self.assertEqual(report.latest_states()[key],(guard,adopted))

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
            observations=[dict(model='7b', dataset='alpaca', measurement_host='C', system='distserve', rate_rps=3,
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

    def test_metric_supplement_does_not_change_historical_slo(self):
        common=dict(model='32b',dataset='alpaca',measurement_host='B',system='distserve',rate_rps=4.5)
        result=dict(complete=False,metric_audit_errors=[],groups=[dict(model='32b',dataset='alpaca',node='B',
            rate_step_rps=.5,rate_grid=[4.5],cap_rate_rps=4.5,complete=False,
            decision=dict(phase='baselines',cap_rate_rps=4.5))],observations=[
                dict(common,cell_id='old1',repeat=1,slo_attainment=.05,energy_j=100,token_throughput_is_exact=False),
                dict(common,cell_id='old2',repeat=2,slo_attainment=.03,energy_j=120,token_throughput_is_exact=False),
                dict(common,cell_id='supplement',repeat=1,measurement_purpose='metric_supplement',slo_attainment=.99,
                     energy_j=1,generated_token_throughput_tps=250,token_throughput_is_exact=True)])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);row=report.export(result,out)[0]
            self.assertEqual(row['repeats'],2);self.assertEqual(row['measurement_count'],3)
            self.assertAlmostEqual(row['slo_attainment_mean'],.04)
            self.assertEqual(row['slo_attainment_cell_ids'],'old1|old2')
            self.assertEqual(row['energy_j_mean'],110)
            self.assertEqual(row['generated_token_throughput_tps_mean'],250)
            self.assertEqual(row['generated_token_throughput_tps_source'],'metric_supplement')
            with (out/'measurements.csv').open() as stream: detail=list(csv.DictReader(stream))
            self.assertEqual(len(detail),3)
            self.assertEqual(detail[-1]['slo_attainment'],'0.99')

    def test_remaining_count_keeps_unknown_caps_and_boundary_repeat(self):
        result=dict(metric_audit_errors=[],observations=[],groups=[
            dict(model='32b',dataset='alpaca',node='B',decision=dict(phase='baselines',baseline_tasks=[
                dict(action='execute',row=dict(measurement_purpose='metric_supplement'))])),
            dict(model='14b',dataset='sharegpt',node='B',decision=dict(phase='pdblend')),
            dict(model='14b',dataset='alpaca',node='Anew20260909',decision=dict(phase='pdblend'))])
        count=report.remaining_work(result)
        self.assertEqual(count['total'],dict(constant=3,variables={'M':5,'N':5},expression='3 + 5M + 5N'))
        a=result['groups'][-1]
        a['decision']=dict(phase='pdblend',rate_rps=3,decision=dict(cap_observed=True))
        for index in range(2):
            result['observations'].append(dict(model='14b',dataset='alpaca',measurement_host='Anew20260909',
                cell_id=str(index),measurement_valid=True,work_complete=True))
        count=report.remaining_work(result)
        self.assertEqual(count['nodes'][-1]['constant'],9)  # 8 baseline cells and one boundary repeat.
        self.assertEqual(count['total']['expression'],'11 + 5M')

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
            root=Path(directory); status=root/'uniform-rate-20260909-v2/restore/status.json'
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

    def test_five_system_scope_requires_all_groups_and_terminal_supervisors(self):
        groups = [dict(model=m, dataset=d, node=c.host(m,d), complete=True, pdb_boundary_complete=True)
                  for m in c.MODELS for d in c.DATASETS]
        result = dict(complete=True, groups=groups)
        states = {(g['model'],g['dataset'],g['node']):(Path('/tmp')/g['model']/g['dataset']/'status.json',
                  dict(complete=True,finished_s=3,node_lease_held=False)) for g in groups}
        self.assertTrue(monitor.completion_state(result,'five_systems',states=states)['scope_complete'])
        self.assertFalse(monitor.completion_state(result,'pdblend',states=states)['scope_complete'])
        self.assertFalse(monitor.completion_state(result,'five_systems',states=states,hydration_active=True)['scope_complete'])
        states[('14b','sharegpt','B')][1]['complete'] = False
        self.assertFalse(monitor.completion_state(result,'five_systems',states=states)['scope_complete'])
        result['complete'] = False
        self.assertFalse(monitor.completion_state(result,'five_systems',states={})['scope_complete'])

    def test_host_pairing_and_per_metric_sample_counts(self):
        group = dict(model='14b',dataset='sharegpt',node='B',rate_step_rps=.25,rate_grid=[.25],
            cap_rate_rps=.25,complete=False,decision=dict(cap_rate_rps=.25))
        row = dict(model='14b',dataset='sharegpt',system='mixed',rate_rps=.25,repeat=1,
            measurement_host='B',cell_id='b1',energy_j=100,slo_attainment=.8,work_complete=False,
            token_throughput_is_exact=False,generated_token_throughput_tps=0)
        complete = dict(row,cell_id='b2',repeat=2,token_throughput_is_exact=True,generated_token_throughput_tps=20)
        other = dict(complete,measurement_host='Anew20260909',cell_id='a1',energy_j=900)
        result = dict(complete=False,metric_audit_errors=[],groups=[group],observations=[row,complete,other])
        with tempfile.TemporaryDirectory() as directory:
            rows = report.export(result,Path(directory))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['measurement_host'],'B')
            self.assertEqual(rows[0]['energy_j_mean'],100)
            self.assertEqual(rows[0]['generated_token_throughput_tps_mean'],20)
            self.assertEqual(rows[0]['generated_token_throughput_tps_n'],1)
            self.assertEqual(rows[0]['energy_j_n'],2)

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
        self.assertEqual(list(monitor.references(dict(schema='uniform-v2-reuse-raw-audit',observations=[value]))),[])

    def test_formal_mirror_does_not_wait_for_qualification_archive(self):
        cp = dict(path='/root/workspace/checkpoint.json', sha256='a'*64)
        gate = dict(path='/root/workspace/large-qualification.json', sha256='b'*64)
        observed = dict(checkpoint=cp, qualification=gate)
        self.assertEqual(list(monitor.scientific_references(observed)), [cp])
        diagnosis = dict(path='/root/workspace/capacity-diagnosis.json', sha256='d'*64)
        observed['diagnosis_reference'] = diagnosis
        self.assertEqual(list(monitor.scientific_references(observed)), [cp, diagnosis])
        raw = '/root/workspace/bench.csv'
        self.assertIn(dict(path=raw,sha256='c'*64),
            list(monitor.scientific_references(dict(artifacts={raw:'c'*64},qualification=gate))))

    def test_dynamic_transition_artifacts_are_mirrored(self):
        path = '/root/workspace/transitions/gpu5-removal.json'
        reference = dict(path=path, sha256='d'*64)
        receipt = dict(dynamic_artifacts={path:'d'*64})
        self.assertEqual(list(monitor.references(receipt)), [reference])
        self.assertEqual(list(monitor.scientific_references(receipt)), [reference])

    def test_failed_nested_transfer_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); leaf=root/'leaf.csv'; leaf.write_text('power')
            leaf_ref=c.ref(leaf); leaf.unlink()
            parent=root/'checkpoint.json'; parent.write_text(json.dumps(dict(artifacts={str(leaf):leaf_ref['sha256']})))
            mirror=monitor.Mirror(root/'cache.json')
            with patch.object(monitor,'safe_path',side_effect=Path), patch.object(mirror,'fetch',side_effect=RuntimeError('temporary link failure')):
                mirror.hydrate('C',[c.ref(parent)])
            self.assertTrue(mirror.retry_refs)
            def transfer(node, refs):
                self.assertEqual(refs,[leaf_ref]);leaf.write_text('power')
            with patch.object(monitor,'safe_path',side_effect=Path), patch.object(mirror,'fetch',side_effect=transfer):
                mirror.hydrate('C',[c.ref(parent)])
            self.assertFalse(mirror.retry_refs)

if __name__=='__main__': unittest.main()
