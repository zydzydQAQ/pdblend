import unittest
from startup_ports_guard_v1 import Guard,parse,encode
PORTS=set(range(37000,37008))|set(range(56000,56225,32))
class Tests(unittest.TestCase):
 def context(self,initial):
  state=[initial];writes=[]
  def read():return state[0]
  def write(v):state[0]=v.strip();writes.append(v)
  return state,writes,Guard(PORTS,read,write)
 def test_empty(self):
  s,w,g=self.context('')
  with g:self.assertEqual(parse(s[0]),PORTS)
  self.assertEqual(s[0],'');self.assertTrue(g.state['restored'])
 def test_existing_ranges_preserved(self):
  s,w,g=self.context('10-30,37001,65535')
  with g:self.assertEqual(parse(s[0]),PORTS|set(range(10,31))|{65535})
  self.assertEqual(s[0],'10-30,37001,65535')
 def test_body_failure_restores(self):
  s,w,g=self.context('1234')
  with self.assertRaises(RuntimeError):
   with g:raise RuntimeError('failed engine startup')
  self.assertEqual(s[0],'1234');self.assertTrue(g.state['restored'])
 def test_cancel_restores(self):
  import asyncio
  s,w,g=self.context('1234')
  with self.assertRaises(asyncio.CancelledError):
   with g:raise asyncio.CancelledError()
  self.assertEqual(s[0],'1234')
 def test_concurrent_change_refused(self):
  s,w,g=self.context('')
  with self.assertRaises(AssertionError):
   with g:s[0]='8888'
  self.assertEqual(s[0],'8888');self.assertFalse(g.state['restored'])
 def test_round_trip(self):self.assertEqual(parse(encode(PORTS)),PORTS)
if __name__=='__main__':unittest.main()
