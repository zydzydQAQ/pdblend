import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent));import restore

def sample():
 b={k:{'value':k} for k in ('Id','Image','Config','HostConfig','Path','Args')};b.update(Mounts=[{'Destination':'/a','Source':'/x','RW':True},{'Destination':'/b','Source':'/y','RW':False}],State={'Running':True,'Pid':1,'StartedAt':'old'});a=copy.deepcopy(b);a['State'].update(Pid=2,StartedAt='new');a['Mounts'].reverse();return b,a

def test_only_order_changes_passes():restore.preserve(*sample())
@pytest.mark.parametrize('bad',['mount_source','mount_rw','duplicate','config','same_pid'])
def test_material_changes_rejected(bad):
 b,a=sample()
 if bad=='mount_source':a['Mounts'][0]['Source']='/foreign'
 if bad=='mount_rw':a['Mounts'][0]['RW']=True
 if bad=='duplicate':a['Mounts'].append(a['Mounts'][0])
 if bad=='config':a['Config']['value']='different'
 if bad=='same_pid':a['State']['Pid']=1
 with pytest.raises(RuntimeError):restore.preserve(b,a)
