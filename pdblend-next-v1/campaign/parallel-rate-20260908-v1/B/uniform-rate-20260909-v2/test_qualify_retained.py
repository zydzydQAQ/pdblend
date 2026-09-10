"""Negative CPU tests against actual retained B evidence, with no GPU calls."""
import copy
from unittest import TestCase, main, mock
import qualify_retained as q


class RetainedEvidence(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = q.pinned(q.PARENT)
        cls.bootstrap = q.read(q.GATE / 'bootstrap.json')
        cls.inventory = q.ref(q.RESTORE / 'containers.after.json')
        cls.binder, _, _, _, cls.power, _ = q.BASE.sources()

    def test_current_binding_restarted_all_four_original_tp2_instances(self):
        q.same_policy(self.original['instances'], self.bootstrap['instances'])
        self.assertTrue(all(a['container']['StartedAt'] != b['container']['StartedAt']
                            for a, b in zip(self.original['instances'], self.bootstrap['instances'])))

    def test_changed_native_source_or_geometry_is_rejected(self):
        altered = copy.deepcopy(self.bootstrap['instances'])
        altered[0]['tp'] = 1
        with self.assertRaisesRegex(RuntimeError, 'policy changed'):
            q.same_policy(self.original['instances'], altered)

    def test_cold_inventory_must_actually_be_stopped(self):
        original_read = q.read
        def changed(path):
            result = original_read(path)
            if str(path).endswith('/containers.before.json'):
                result[0]['State']['Running'] = True
            return result
        with mock.patch.object(q, 'read', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'target was not stopped'):
                q.restore_contract(self.bootstrap, self.original, self.inventory, self.binder, self.power)

    def test_retained_container_arguments_cannot_change(self):
        original_read = q.read
        def changed(path):
            result = original_read(path)
            if str(path).endswith('/containers.before.json'):
                result[0]['Args'].append('--different-policy')
            return result
        with mock.patch.object(q, 'read', side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, 'stopped retained execution changed: Args'):
                q.restore_contract(self.bootstrap, self.original, self.inventory, self.binder, self.power)


if __name__ == '__main__':
    main()
