"""Publish only a verified local 100-second experiment index; no engine controls."""
import pathlib,json,hashlib,time,socket,copy
R=pathlib.Path('/root/workspace/pdblend-next-v1'); C=R/'campaign'; P=C/'B32B-five-system100-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,b):
 t=p.with_suffix(p.suffix+'.tmp');t.write_bytes(b);t.replace(p)
def main():
 binding=P/'binding.pdblend.r2.json'; b=read(binding); source=C/'five-system-fixed-window-v1/sources/B32B/manifest.json'; spec=read(source)
 assert socket.gethostname()==b['hostname'] and sha(source)==b['files'][str(source)]
 bridgepath=P/'bridge-pdblend-r2/status.json'; bridge=read(bridgepath); assert bridge['phase'] in ('main','scale')
 processes={}
 for label,key in [('bridge','pid'),('runner','child_pid')]:
  pid=bridge[key];cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert '--binding '+str(binding) in cmd
  processes[label]={'pid':pid,'command':cmd}
 rows={r['cell_id']:r for r in spec['cells'] if r['system']=='pdblend'};actual=[]
 for path in pathlib.Path(b['output']).glob('cells/*/runtime_config.json'):
  row=rows[path.parent.name];cfg=read(b['configs'][row['dataset']]);assert sha(b['configs'][row['dataset']])==b['files'][b['configs'][row['dataset']]]
  cfg.update(journal=str(path.parent/'control.jsonl'),slo_scale=row['slo_scale'],slo_protocol='per-dataset-slo-v1',slo_attainment_target=.9,slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s'],comparison_system=row['system'])
  assert read(path)==cfg;actual.append({'path':str(path),'sha256':sha(path),'cell_id':row['cell_id']})
 assert actual,'no actual new controller configuration yet'
 previous=C/'current-experiment.json';old=read(previous);record=copy.deepcopy(old)
 record.update(schema=2,written_s=time.time(),protocol_id=spec['protocol_id'],arrival_seeds=[701],arrival_window_seconds=100,minimum_requests=None,main_cells=150,additional_scale_cells=90,scale_one_references_reused=45,baseline_execution='new matched measurements explicitly authorized; legacy deployment pending',baseline_policy='All five systems share each exact new 100s trace; historical results remain read-only',phase_ledgers={'previous_read_only':old.get('phase_ledgers'),'new_invocations':str(pathlib.Path(b['output'])/'invocations')})
 record['package']={'path':str(P),'binding':str(binding),'binding_sha256':sha(binding),'source_manifest':str(source),'source_manifest_sha256':sha(source),'runner':str(C/'five-system-execution-v2/run.py'),'runner_sha256':bridge['runner_sha256']}
 record['controller'].update(host_release=b['host_release'],configs={d:{'path':p,'sha256':sha(p)} for d,p in b['configs'].items()})
 record['controller'].pop('path',None);record['controller'].pop('sha256',None)
 record['current_system']='pdblend';record['current_system_main_cells']=30;record['current_system_additional_scale_cells']=18;record['comparison_systems']=spec['comparison_systems']
 record['bridge']={'path':str(bridgepath),'phase_at_verification':bridge['phase'],'processes':processes};record['deadline']={'global_end_seconds':b['deadline_s'],'global_end_cst':'2026-09-08 21:06:10 CST','per_cell_window_s':100,'request_hard_timeout_s':120,'drain_allowance_s':120,'outer_cleanup_allowance_s':90,'previous_phase_limits_superseded_by_user100s_protocol':True}
 record['actual_configuration_verification']={'verified_s':time.time(),'count':len(actual),'records':actual};record['updater']={'path':str(pathlib.Path(__file__).resolve()),'sha256':sha(__file__)}
 history=C/'current-experiment-history'/f'{time.time_ns()}-32b-five-system100';history.mkdir(parents=True,exist_ok=False)
 for src,name in [(previous,'previous.json'),(C/'CURRENT_EXPERIMENT.md','previous.md')]:
  if src.exists():assert src.is_file() and not src.is_symlink();(history/name).write_bytes(src.read_bytes())
 data=(json.dumps(record,indent=2,ensure_ascii=False)+'\n').encode();(history/'current.json').write_bytes(data);atomic(previous,data);assert previous.read_bytes()==data
 lines=['# B 本机当前实验：Qwen2.5-32B-Instruct','', '本入口指向实际冻结配置，不覆盖运行中服务参数。','', '新比较：PDBlend、Mixed、DistServe、EcoServe、DynamoLLM resident；3 个数据集各 10 个 rate，种子 701，每格固定 100 秒。共 150 主格，另有 90 格 SLO 0.5×/2×，1×复用各系统主结果。', '', '当前运行 PDBlend 两个 TP2、8192/32、uniform prior 211；其他四系统需切换到各自原实现与四个 TP2，不能同时发压。DynamoLLM resident 不表示物理扩缩容已实现。','', 'Alpaca TTFT/TPOT 1s/0.1s；ShareGPT 5s/0.15s；LongBench 15s/0.2s；联合目标 90%。请求硬超时 120s、到达窗后排空上限 120s。全 8 卡原始能量含闲置和失败，主能量与外围操作能量分列。', '', f'实际绑定：{binding}', f'实际结果：{b["output"]}',f'当前桥 PID：{processes["bridge"]["pid"]}；阶段：{bridge["phase"]}。', '总截止：2026-09-08 21:06:10 CST。旧 300 秒结果与失败尝试完整保留。',f'完整实际 Controller 配置已核对 {len(actual)} 份。',f'JSON SHA256：{sha(previous)}','']
 md='\n'.join(lines).encode();(history/'CURRENT_EXPERIMENT.md').write_bytes(md);atomic(C/'CURRENT_EXPERIMENT.md',md)
 receipt={'hostname':socket.gethostname(),'model':'32b','json_sha256':sha(previous),'markdown_sha256':sha(C/'CURRENT_EXPERIMENT.md'),'history':str(history),'actual_configurations_verified':len(actual),'gpu_controls_sent':False,'active_frozen_files_changed':False}
 (history/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(P/'current-index-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
