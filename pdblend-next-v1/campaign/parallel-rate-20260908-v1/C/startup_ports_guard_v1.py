"""Temporarily exclude original service ports from ephemeral source allocation.

The caller must hold the unique node lease. No engine configuration changes.
"""
from pathlib import Path
import time
PATH=Path('/proc/sys/net/ipv4/ip_local_reserved_ports')
def parse(value):
 ports=set()
 for token in value.strip().split(','):
  if not token:continue
  pair=token.split('-');assert len(pair) in (1,2)
  a=int(pair[0]);b=int(pair[-1]);assert 1<=a<=b<=65535;ports.update(range(a,b+1))
 return ports
def encode(ports):
 groups=[]
 for n in sorted(ports):
  if groups and n==groups[-1][1]+1:groups[-1][1]=n
  else:groups.append([n,n])
 return ','.join(str(a) if a==b else f'{a}-{b}' for a,b in groups)
class Guard:
 def __init__(self,ports,read=None,write=None):
  self.ports=set(ports);assert len(self.ports)==16 and all(type(p) is int and 1<=p<=65535 for p in self.ports)
  self.read=read or PATH.read_text;self.write=write or PATH.write_text;self.state=dict(schema='original-C16-startup-port-reservation-v1',service_ports=sorted(self.ports),complete=False,restored=False)
 def __enter__(self):
  self.before=self.read().strip();self.applied=encode(parse(self.before)|self.ports);self.state.update(started_s=time.time(),before=self.before,applied=self.applied)
  try:
   self.write(self.applied+'\n');actual=self.read().strip();assert parse(actual)==parse(self.applied),'kernel reservation not applied';self.state['applied_observed']=actual;return self
  except BaseException:
   self.write(self.before+'\n');self.state['after']=self.read().strip();self.state['restored']=self.state['after']==self.before;raise
 def __exit__(self,kind,value,tb):
  try:
   current=self.read().strip();assert parse(current)==parse(self.applied),'concurrent reserved-port mutation while node lease held'
   self.write(self.before+'\n');after=self.read().strip();self.state['after']=after;assert after==self.before,'original reserved ports not precisely restored';self.state.update(restored=True,complete=True)
  except BaseException as exc:self.state['error']=repr(exc);raise
  finally:self.state['finished_s']=time.time()
