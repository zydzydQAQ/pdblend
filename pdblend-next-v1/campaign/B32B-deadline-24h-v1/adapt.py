"""Build local new wrapper from byte-frozen predecessor; never edits predecessor."""
from pathlib import Path
P=Path(__file__).resolve().parent
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def main():
 p=P/'run.py';s=p.read_text().replace("releases/io-v1.2.1-runtime","releases/io-v1.2.1-fixed-window-decode8-v1-runtime")
 start=s.index("    original=read(ROOT/'inputs/controller.source.json')");end=s.index('    return spec',start)
 s=s[:start]+'''    original=read(ROOT/'inputs/controller.source.json');original.update(controller_source_release=str(HOST),
        measurement_window_protocol='per-dataset-slo-fixed-window-v2',arrival_window_s=300,slo_scale=1,
        profiles=str(ROOT.parent/'B32B-decode8-composite-candidate-v1/profiles.json'))
    require(cfg==original,'policy differs from declared uniform fixed-window/decode-phase candidate')
    require(not cfg.get('prediction_cap_to_max_tokens') and not cfg.get('output_limit_aware_prediction')
        and cfg['output_prior']==211,'uniform prior/estimator differs')
    import fixed_queue
    fixed_queue.validate_spec(sys.modules[__name__],spec)
'''+s[end:]
 start=s.index("    anchors_path=ROOT.parent/");end=s.index("    gates=read(ROOT/'inputs/correctness-source.json')",start);s=s[:start]+s[end:]
 s=replace(s,"    return dict(files=paths,large_inputs=", "    for row in read(ROOT/'runspec.json')['cells']:add(row['trace'])\n    add(ROOT.parent/'deadline-24h-v1/protocol.json')\n    return dict(files=paths,large_inputs=")
 start=s.index('async def run(part):');end=s.index('\n\nif __name__',start)
 s=s[:start]+'''async def cli(args):
    task=asyncio.current_task();interrupted=False
    def stop():
        nonlocal interrupted
        if not interrupted:interrupted=True;task.cancel()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,stop)
    if args.action=='prepare':await prepare()
    else:
        import fixed_queue
        from execution import Cell
        await fixed_queue.sweep(sys.modules[__name__],Cell,phase=args.phase,max_cells=args.max_cells)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('check','prepare','run'),default='check')
    p.add_argument('--phase',choices=('probe','main','scale'),default='probe');p.add_argument('--max-cells',type=int,default=1);args=p.parse_args()
    if args.action=='check':package_check();print(json.dumps(dict(package_valid=True,cells=len(read(ROOT/'runspec.json')['cells']),gpu_executed=False)));return
    from ecopadg.serving.campaign import node_lease
    with node_lease():asyncio.run(cli(args))
'''+s[end:];p.write_text(s)
 p=P/'execution.py';s=p.read_text();s=replace(s,'def __init__(self,b,session,row,freeze,status,status_name,hardware):','def __init__(self,b,session,row,freeze,status,status_name,hardware,*,limits):')
 s=replace(s,'        self.status,self.status_name,self.hardware=status,status_name,hardware','        self.status,self.status_name,self.hardware=status,status_name,hardware\n        self.limits=limits;self.b.require(limits["arrival_window_s"]==300,"wrong limits")')
 s=replace(s,'        status[\'cells\'].append(self.receipt);self.save()','        self.receipt["limits"]=limits\n        b.write("operations/"+row["cell_id"]+"/limits.json",limits)\n        status[\'cells\'].append(self.receipt);self.save()')
 s=replace(s,"            start=self.receipt['operation_start_s']=time.time();self.mutated=True","            b.require(time.time()<=self.limits['latest_arrival_epoch_s'],'startup time exhausted before controls')\n            start=self.receipt['operation_start_s']=time.time();self.mutated=True")
 s=replace(s,"            self.child_log=(self.operation/'child.log').open('xb')","            b.require(time.time()<=self.limits['latest_arrival_epoch_s'],'startup time exhausted before child')\n            self.child_log=(self.operation/'child.log').open('xb')")
 s=replace(s,"            until=time.monotonic()+self.row['trace_duration_s']+330","            until=self.limits['cell_execution_deadline_s']")
 s=replace(s,"                    b.require(time.monotonic()<until,'host exceeded trace plus bounded drain window')","                    b.require(not sampler.error,'outer instantaneous power failed')\n                    b.require(time.time()<until,'host exceeded fixed-window execution deadline')")
 s=replace(s,"            expected_config.update({k:self.row[k] for k in ('slo_ttft_s','slo_tpot_s')})","            expected_config.update({k:self.row[k] for k in ('slo_ttft_s','slo_tpot_s','slo_scale')})\n            expected_config.update(slo_protocol='per-dataset-slo-v1',slo_attainment_target=.9)")
 s=replace(s,"            self.receipt['effective_config_sha256']=b.sha(self.out/'runtime_config.json')","            from evidence import validate_window\n            validate_window(b,self.row,summary,b.read(self.operation/'actual_epoch_gate.json'),self.limits,self.out)\n            self.receipt['effective_config_sha256']=b.sha(self.out/'runtime_config.json')")
 s=replace(s,"            try:cleanup=await self.cleanup();self.receipt['outer_cleanup']=cleanup","            try:\n                remaining=min(90,max(.01,self.limits['restore_deadline_s']-time.time()-8))\n                cleanup=await asyncio.wait_for(self.cleanup(),remaining);self.receipt['outer_cleanup']=cleanup")
 # Pure schema/cleanup failures stop the queue; SLO and completed-work failures remain measurements.
 start=s.index('\n\nasync def sweep(');s=s[:start]+'\n';p.write_text(s)
 p=P/'child.py';s=p.read_text().replace('releases/io-v1.2.1-runtime','releases/io-v1.2.1-fixed-window-decode8-v1-runtime')
 s=replace(s,"slo_tpot_s=row['slo_tpot_s'],timeout=120)","slo_tpot_s=row['slo_tpot_s'],slo_scale=row['slo_scale'],timeout=120)")
 s=replace(s,"    original=aiohttp.ClientSession._request","    import ecopadg.serving.cell as cell_module\n    from epoch import EpochGuard\n    limits=json.loads((operation/'limits.json').read_text())\n    benchmark=cell_module.load_benchmark()\n    original_headers=benchmark.evaluation_headers\n    guard=EpochGuard(limits,operation/'actual_epoch_gate.json',original_headers)\n    benchmark.evaluation_headers=guard\n    original=aiohttp.ClientSession._request")
 s=replace(s,"        finally:aiohttp.ClientSession._request=original","        finally:\n            aiohttp.ClientSession._request=original\n            benchmark.evaluation_headers=original_headers")
 p.write_text(s)
if __name__=='__main__':main()
