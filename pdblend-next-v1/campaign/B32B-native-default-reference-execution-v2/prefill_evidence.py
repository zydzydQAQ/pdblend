"""CPU-only validation after HTTP, diagnostic container and workers terminate."""
import json
import shutil
from pathlib import Path


def require(ok, message):
    if not ok: raise RuntimeError(message)


def verify(spec, out):
    from adapter import read, sha
    out = Path(out)
    cfg = read(spec['diagnostic_instance']['config'])
    runtime = Path(cfg['runtime_dir'])
    configs = read(runtime/'worker-config.json')
    for phase in ('before','after'):
        rows = configs[phase]
        require(len(rows)==2 and {r['rank'] for r in rows}=={0,1}, 'two actual worker config ranks')
        for r in rows:
            require(all(r[k] is True for k in ('runner_chunked','builder_chunked','builder_config_chunked',
                'runner_builder_config_same')) and r['tokens']==8192 and r['seqs']==32, 'all worker configs remain original True/8192/32')
    require(sorted(configs['before'],key=lambda r:r['rank']) == sorted(configs['after'],key=lambda r:r['rank']), 'same actual workers/configs before and after')
    declaration = read(spec['observation_spec'])
    requests = {r['request_uuid']: r['body']['prompt'] for r in declaration['requests']}
    source = Path(declaration['output_dir']).parent/'prefill-capture-live'
    files = sorted(source.iterdir())
    require(len(files)==4 and all(p.is_file() and not p.is_symlink() and p.stat().st_size<=65536 for p in files), 'two logs/two statuses only bounded files')
    logs = [p for p in files if p.suffix=='.jsonl']
    statuses = [p for p in files if p.name.endswith('.status.json')]
    require(len(logs)==len(statuses)==2, 'two prefill rank files')
    by_rank = {}
    for log in logs:
        raw = log.read_bytes();require(raw.endswith(b'\n'), 'prefill log terminal newline')
        rows = [json.loads(x) for x in raw.splitlines()]
        require(len(rows)==4 and len({r['request_id'] for r in rows})==4, 'four distinct prefills per rank')
        rank = rows[0]['rank'];require(rank in (0,1) and rank not in by_rank, 'two unique ranks')
        for r in rows:
            rid = r['request_id'];n=len(requests[rid]);m=r['metadata']
            require(r['rank']==rank and r['spec_sha256']==sha(spec['observation_spec']) and
                r['pid']==configs['before'][next(i for i,x in enumerate(configs['before']) if x['rank']==rank)]['pid'], 'prefill spec/rank/process binding')
            require(m['input_tokens']==requests[rid] and m['positions']==list(range(n)) and m['seq_lens']==[n]
                and m['context_lens']==[0] and m['query_start_loc']==[0,n], 'actual full prompt metadata')
            require(r['num_prefills']==1 and r['num_decode_tokens']==0 and r['num_prefill_tokens']==n,
                'one pure full prefill')
            table=m['block_tables'][0]
            require(len(m['block_tables'])==1 and len(table)>=n//16 and r['block_tables_shape']==[1,len(table)] and
                r['block_tables_numel']==len(table)>0 and m['slots']==[table[i//16]*16+i%16 for i in range(n)], 'actual block/slot mapping')
            require(len(r['kv_cache_numel_per_layer'])==64 and all(type(v) is int and v>0 for v in r['kv_cache_numel_per_layer'])
                and r['inferred_flash_prefill_branch']=='paged_kv' and r['branch_is_source_inference'] is True,
                'frozen source inferred actual prefill predicate')
            require(r['worker']==next(x for x in configs['before'] if x['rank']==rank) and
                r['before_forward_s']<=r['after_sampler_read_s'], 'actual worker/time evidence')
        by_rank[rank]={r['request_id']:r for r in rows}
        st=read(log.with_suffix('.status.json'))
        require(st['rank']==rank and st['pid']==rows[0]['pid'] and st['complete'] is True and st['failed'] is False and
            st['written']==st['expected_records']==4,'four writer records, externally terminal')
    require(set(by_rank)=={0,1},'both ranks')
    for rid in requests:
        require(by_rank[0][rid]['metadata']==by_rank[1][rid]['metadata'],'TP prefill metadata parity')
    target=out/'prefill-capture-frozen';target.mkdir()
    for p in files:shutil.copyfile(p,target/p.name)
    shutil.copyfile(runtime/'worker-config.json',out/'worker-config.json')
    return dict(complete=True,records=8,requests=4,ranks=2,all_worker_configs_true=True,
        branch='paged_kv',branch_is_source_inference=True,
        files={str(p):sha(p) for p in target.iterdir()},worker_config_sha256=sha(out/'worker-config.json'),
        old003_prefill_was_not_captured=True)
