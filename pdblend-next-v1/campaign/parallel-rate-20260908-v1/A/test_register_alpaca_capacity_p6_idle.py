"""Read-only regressions for actual phase/owner mapping before idle evidence union."""
import copy
import unittest
from pathlib import Path
import register_alpaca_capacity_p6_idle as register


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=register.terminal_stage(register.FULL,27)
        cls.new=register.terminal_stage(register.IDLE,2)
        cls.identity=cls.old['capacity']['identity']

    def test_all_six_selected_actual_idle_windows_pass_original_verifier(self):
        for stage,cycle in [(self.old,1),(self.old,3),(self.new,1)]:
            for layout in (2,3):
                phase=f'cycle-{cycle}-idle-layout{layout}'
                reference=register.ref(Path(stage['output'])/phase/'result.json')
                result=register.verify_idle_mapping(reference,self.identity,stage,phase)
                self.assertGreater(result['_idle_w'],0)

    def test_rejected_original_idle_remains_rejected(self):
        phase='cycle-2-idle-layout3';reference=register.ref(register.FULL/phase/'result.json')
        with self.assertRaisesRegex(ValueError,'continuously cover'):
            register.verify_idle_mapping(reference,self.identity,self.old,phase)

    def test_cross_owner_and_wrong_phase_cannot_use_union_to_pass(self):
        phase='cycle-1-idle-layout3';reference=register.ref(register.FULL/phase/'result.json')
        with self.assertRaises(ValueError):register.verify_idle_mapping(reference,self.identity,self.new,phase)
        with self.assertRaises(ValueError):register.verify_idle_mapping(reference,self.identity,self.old,'cycle-3-idle-layout3')
        changed=copy.deepcopy(self.old);changed['capacity']['owner_id']='foreign-owner'
        with self.assertRaises(ValueError):register.verify_idle_mapping(reference,self.identity,changed,phase)

    def test_wrong_owner_pid_and_measurement_window_rejected(self):
        phase='cycle-1-idle-layout3';reference=register.ref(register.FULL/phase/'result.json')
        changed=copy.deepcopy(self.old);changed['inventory']['pid']+=1
        with self.assertRaises(ValueError):register.verify_idle_mapping(reference,self.identity,changed,phase)
        changed=copy.deepcopy(self.old);changed['full_operation']['measurement_start_s']=changed['status']['finished_s']
        with self.assertRaises(ValueError):register.verify_idle_mapping(reference,self.identity,changed,phase)

    def test_actual_union_retains_both_sources_without_physical_snapshot_claim(self):
        union=register.inventory_union([self.old,self.new],self.identity,[])
        self.assertFalse(union['physical_runtime_inventory'])
        self.assertFalse(union['formal_checkpoint_eligible'])
        self.assertEqual(len(union['source_inventories']),2)
        self.assertEqual(len(union['known_instances']),6)

    def test_union_physical_identity_conflict_rejected(self):
        changed=copy.deepcopy(self.new);changed['inventory']['known_instances']['nextv3a6']['host_pid']+=1
        with self.assertRaises(ValueError):register.inventory_union([self.old,changed],self.identity,[])
        changed=copy.deepcopy(self.new)
        extra=next(iid for iid in self.old['inventory']['known_instances'] if iid not in self.old['inventory']['initial_ids'])
        changed['inventory']['known_instances'][extra]=copy.deepcopy(self.old['inventory']['known_instances'][extra])
        changed['inventory']['known_instances'][extra]['owner_id']='foreign-owner'
        with self.assertRaises(ValueError):register.inventory_union([self.old,changed],self.identity,[])


if __name__=='__main__':unittest.main()
