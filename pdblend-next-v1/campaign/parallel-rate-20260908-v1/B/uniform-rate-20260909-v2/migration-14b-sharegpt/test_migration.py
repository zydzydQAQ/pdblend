"""CPU negative checks for fresh-host qualification and automatic hardware handoff."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import qualify_fixed as q
import power_selftest as power
import pipeline


class Qualification(unittest.TestCase):
    def test_entire_fresh_shape_set(self):
        profile=json.loads((power.ROOT/'B/distributed-14b-v1/frequency2100-registered-001/profiles.development.json').read_text())
        shapes=q.shapes(profile)
        self.assertEqual(len(shapes)*2,82)
        self.assertEqual({s[0] for s in shapes},{900,1500,2100})
        self.assertEqual({s[1] for s in shapes},{128,2048,6144,7168})
        self.assertNotIn('partial',Path(q.__file__).read_text())

    def test_loaded_clock_rejects_outside_fifteen(self):
        rows=[(t,[2100]*8) for t in (0,.1,.2,.3,.4)]
        self.assertTrue(q.clock_window(rows,[6],2100,0,.4)['passed'])
        rows[2][1][6]=2116
        with self.assertRaises(AssertionError):q.clock_window(rows,[6],2100,0,.4)

    def test_loaded_clock_rejects_sampling_gap(self):
        with self.assertRaises(AssertionError):q.clock_window([(0,[2100]*8),(.4,[2100]*8)],[6],2100,0,.4)

    def test_identity_rejects_A_or_reordered_GPU(self):
        identity=power.read(power.IDENTITY)
        self.assertIsNone(power.validate_identity(identity,'iZwz9i5bte3xkpmcoes3t2Z',identity['GPUs']))
        for hostname,rows in [('iZwz9274emxme9019d2sjgZ',identity['GPUs']),('iZwz9i5bte3xkpmcoes3t2Z',list(reversed(identity['GPUs'])))]:
            with self.assertRaises(AssertionError):power.validate_identity(identity,hostname,rows)

    def test_stream_requires_all_token_evidence(self):
        tokens=list(range(64));row=dict(success=True,http_status=200,done_marker=True,
            prompt_token_ids=([9707,1879,13]*43)[:128],output_token_ids=tokens,
            usage=dict(prompt_tokens=128,completion_tokens=64),token_received_s=list(range(64)))
        q.stream_check(row,128,tokens)
        row['output_token_ids']=tokens[:-1]
        with self.assertRaises(AssertionError):q.stream_check(row,128,tokens)


class Handoff(unittest.TestCase):
    def setUp(self):
        self.saved=dict(complete=True,finished_s=4,node_lease_held=False,node='B',model='32b',scope='five_systems',pid=1,
            last_cell_status={'path':'last'},declaration={'path':'declaration'},observations=[])
        self.last=dict(complete=True,finished_s=3,node_lease_held=False,failed=[],pid=2,release={'path':'release'})
        self.release=dict(binding={'path':'binding'})
        self.binding=dict(model='32b',hostname='iZwz9i5bte3xkpmcoes3t2Z')
    def checked(self,r):return {'last':self.last,'release':self.release,'binding':self.binding}[r['path']]
    def run_check(self):
        with patch.object(pipeline.p,'checked',side_effect=self.checked),patch.object(pipeline.p,'active_owner',return_value=False),\
             patch.object(pipeline.contract,'resolve_group',return_value={}),\
             patch.object(pipeline.contract,'select_group',return_value=dict(phase='complete',five_system_complete=True)):
            return pipeline.check_terminal(self.saved)
    def test_success_terminal_binds_actual_last_system(self):self.assertEqual(self.run_check(),{'path':'binding'})
    def test_incomplete_or_failed_predecessor_rejected(self):
        for key,value in [('complete',False),('node_lease_held',True),('node','C'),('scope','pdblend'),('error','failure')]:
            original=copy.deepcopy(self.saved);self.saved[key]=value
            with self.assertRaises(ValueError):self.run_check()
            self.saved=original
    def test_last_cell_failure_rejected(self):
        self.last['failed']=['bad']
        with self.assertRaises(ValueError):self.run_check()
    def test_live_owner_rejected(self):
        with patch.object(pipeline.p,'active_owner',return_value=True):
            with self.assertRaisesRegex(ValueError,'still running'):pipeline.check_terminal(self.saved)
    def test_incomplete_five_system_group_rejected(self):
        with patch.object(pipeline.p,'checked',side_effect=self.checked),patch.object(pipeline.p,'active_owner',return_value=False),\
             patch.object(pipeline.contract,'resolve_group',return_value={}),\
             patch.object(pipeline.contract,'select_group',return_value=dict(phase='baselines',five_system_complete=False)):
            with self.assertRaisesRegex(ValueError,'scope incomplete'):pipeline.check_terminal(self.saved)

if __name__=='__main__':unittest.main()
