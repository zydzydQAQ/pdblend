import copy
import unittest
from docker_equivalence import mounts_equivalence


class MountTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(Destination='/models',Source='/root/workspace/models',Type='bind',Mode='ro',RW=False,Propagation='rprivate'),
                   dict(Destination='/root/workspace',Source='/root/workspace',Type='bind',Mode='',RW=True,Propagation='rprivate')]

    def test_only_order_change_passes(self):
        self.assertTrue(mounts_equivalence(self.rows,self.rows[::-1])['only_array_order_changed'])

    def test_duplicate_destination_rejected(self):
        with self.assertRaises(RuntimeError):mounts_equivalence(self.rows,[self.rows[0],self.rows[0]])

    def test_every_mount_field_remains_exact(self):
        for field,value in [('Source','/other'),('Destination','/other'),('Type','volume'),('Mode','rw'),('RW',True),('Propagation','shared')]:
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.assertRaises(RuntimeError):mounts_equivalence(self.rows,rows)
        rows=copy.deepcopy(self.rows);rows[0]['unreviewed']='field'
        with self.assertRaises(RuntimeError):mounts_equivalence(self.rows,rows)


if __name__=='__main__':unittest.main()
