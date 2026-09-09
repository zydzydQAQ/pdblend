"""Reject a late real benchmark epoch before its first request worker exists."""
import json,math,time
class EpochGuard:
 def __init__(self,limits,path,original):self.limits,self.path,self.original=limits,path,original;self.seen=False
 def __call__(self,api_base,protocol,arrival,dispatch):
  if not self.seen:
   self.seen=True
   valid=(type(arrival) in (int,float) and math.isfinite(arrival) and arrival==dispatch
    and self.limits['issued_s']<=arrival<=self.limits['latest_arrival_epoch_s'])
   with self.path.open('x') as f:json.dump(dict(schema=1,actual_epoch_s=arrival,dispatch_s=dispatch,checked_s=time.time(),
    before_any_request_worker=True,accepted=valid,limits=self.limits),f,indent=2,allow_nan=False)
   if not valid:raise RuntimeError('actual benchmark epoch exceeds admitted startup budget; zero dispatch')
  return self.original(api_base,protocol,arrival,dispatch)
