"""CPU policy and live event-guard boundaries; no HTTP or GPU operations."""
import json
from pathlib import Path
import tempfile
import unittest
import run


class PairTests(unittest.TestCase):
    def test_configs_differ_only_in_token_budget(self):
        a,b=[json.loads((run.ROOT/f'budget{t}.config.json').read_text()) for t in (8192,2048)]
        run.verify_policy(a,8192);run.verify_policy(b,2048)
        a.pop('scheduler_budget_ablation');b.pop('scheduler_budget_ablation');self.assertEqual(a,b)

    def test_pd_or_predictor_or_prior_change_is_rejected(self):
        for key,value in [('allow_pd',True),('output_limit_aware_prediction',True),('output_prior',256)]:
            c=json.loads((run.ROOT/'budget8192.config.json').read_text());c[key]=value
            with self.assertRaises(RuntimeError):run.verify_policy(c,8192)

    def guard(self,event,split=False):
        obj=object.__new__(run.Pair);obj.positions={'one':0};obj.tails={'one':b''};obj.owned={'one':set()};obj.counts={'one':0}
        previous_old,previous_ids=run.OLD,run.IDS
        with tempfile.TemporaryDirectory() as tmp:
            run.OLD=Path(tmp);run.IDS=('one',);(run.OLD/'runtime').mkdir()
            path=run.OLD/'runtime/one.control.events.jsonl'
            raw=json.dumps(event).encode()+b'\n'
            try:
                if split:
                    path.write_bytes(raw[:-2]);obj.poll_events(2048);self.assertEqual(obj.counts['one'],0)
                    with path.open('ab') as f:f.write(raw[-2:])
                else:path.write_bytes(raw)
                obj.poll_events(2048);return obj
            finally:run.OLD=previous_old;run.IDS=previous_ids

    def test_temporal_model_step_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'forbidden temporal'):
            self.guard(dict(mode='temporal',role='mixed',tokens=1,request_ids=['abc']))

    def test_budget_overflow_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'exceeds fixed'):
            self.guard(dict(mode='continuous',role='mixed',tokens=2049,request_ids=['abc']))

    def test_partial_owner_event_is_retained_until_newline(self):
        obj=self.guard(dict(mode='continuous',role='mixed',tokens=2048,request_ids=['abc']),True)
        self.assertEqual(obj.counts['one'],1);self.assertEqual(obj.owned['one'],{'abc'})


if __name__=='__main__':unittest.main(verbosity=2)
