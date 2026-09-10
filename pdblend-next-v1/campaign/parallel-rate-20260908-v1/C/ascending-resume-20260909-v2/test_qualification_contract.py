"""CPU-only negative tests for real clock and legacy-native qualification gates."""
import copy
import unittest
import qualify as q
import cold_restore as c
import docker_equivalence as dns


class ContractTests(unittest.TestCase):
    def test_loaded_clock_rejects_wrong_gpu_and_missing_interval(self):
        samples=[(t,[1500.]*8) for t in (1.,1.1,1.2,1.3)]
        self.assertTrue(q.clock_window(samples,[6],1500,1.,1.3)['passed'])
        broken=copy.deepcopy(samples);broken[1][1][6]=900
        with self.assertRaises(AssertionError):q.clock_window(broken,[6],1500,1.,1.3)
        with self.assertRaises(AssertionError):q.clock_window(samples[::3],[6],1500,1.,1.3)

    def test_legacy_cancel_requires_observed_empty_receive_resources(self):
        row=dict(buffered_tensors=0,buffered_gpu_bytes=0,inflight_receives=0,listener_alive=True,allocations={})
        q.legacy_transfer_rows([row])
        for field in ('buffered_tensors','buffered_gpu_bytes','inflight_receives'):
            with self.assertRaises(AssertionError):q.legacy_transfer_rows([dict(row,**{field:1})])
        with self.assertRaises(AssertionError):q.legacy_transfer_rows([])
        with self.assertRaises(AssertionError):q.legacy_transfer_rows([dict(row,listener_alive=False)])

    def test_retained_launch_config_change_is_not_dns_equivalence(self):
        original=dict(Id='id',Name='/owned',Image='digest',Path='python3',Args=['--config','f'],
                      Config={'Cmd':['python3']},Mounts=[{'Destination':'/models','Source':'/models','RW':False},{'Destination':'/root/workspace','Source':'/root/workspace','RW':True}],HostConfig={'Dns':None,'IpcMode':'host'},
                      State={'Running':False,'Pid':0})
        current=copy.deepcopy(original);current['HostConfig']['Dns']=[]
        self.assertTrue(c.container_equivalence(original,current,dns)['host_config']['equivalent'])
        current['HostConfig']['IpcMode']='private'
        with self.assertRaises(RuntimeError):c.container_equivalence(original,current,dns)
        current=copy.deepcopy(original);current['Args']=['--config','other']
        with self.assertRaises(AssertionError):c.container_equivalence(original,current,dns)


if __name__=='__main__':unittest.main()
