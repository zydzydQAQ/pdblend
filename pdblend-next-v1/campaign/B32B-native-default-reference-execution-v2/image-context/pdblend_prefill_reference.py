"""Opt-in four-prefill metadata observation. No KV contents and no arithmetic edits.

The actual FlashAttention predicate is inferred from its frozen source and the
actual metadata/cache shapes supplied to ModelRunner, not a branch-site trace.
Small metadata clones enqueue before forward; host transfer occurs after sampler.
"""
import json
import os
import time
from pathlib import Path
from vllm import pdblend_diagnostics as old

_STATE = None
_LOADED = False
FLASH_SHA = '2860e916c99fc319ceeab3af8cc11a7d61baea3679ce090724e58087d31ae0b2'


class Writer(old.Writer):
    def submit(self, value):
        if self.submitted >= 4:
            self.failed = True; self.error = 'four-prefill record cap'; return
        super().submit(value)

    def _status(self):
        p = self.root / ('rank%d-pid%d.status.json' % (self.rank, self.pid))
        value = dict(schema=1, rank=self.rank, pid=self.pid, failed=self.failed,
            error=self.error, written=self.written, complete=self.written == 4 and not self.failed,
            expected_records=4, time_s=time.time())
        tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(value, sort_keys=True)+'\n'); tmp.replace(p)


def state():
    global _STATE, _LOADED
    if not _LOADED:
        _LOADED = True
        # A separate explicit opt-in; importing this module never starts a writer.
        if os.environ.get('PDBLEND_PREFILL_REFERENCE') == '1':
            s = old.state()
            if s is not None:
                requests = s.spec.get('requests', [])
                old.require(len(requests) == 4, 'exact four declared prefills')
                _STATE = dict(spec_sha=s.spec_sha,
                    prompts={r['request_uuid']: r['body']['prompt'] for r in requests},
                    root=str(Path(s.spec['output_dir']).parent/'prefill-capture-live'), seen=set(), writer=None)
    return _STATE


def snapshot(worker):
    from vllm.distributed import get_tensor_model_parallel_rank
    r = worker.model_runner
    return flags(r, get_tensor_model_parallel_rank())


def flags(runner, rank):
    c, b = runner.scheduler_config, runner.builder
    return dict(rank=rank, pid=os.getpid(), runner_chunked=c.chunked_prefill_enabled,
        builder_chunked=b.chunked_prefill_enabled,
        builder_config_chunked=b.scheduler_config.chunked_prefill_enabled,
        runner_builder_config_same=c is b.scheduler_config,
        tokens=c.max_num_batched_tokens, seqs=c.max_num_seqs)


def begin(model_input, runner, rank, kv_caches):
    s = state()
    if s is None: return None
    a = model_input.attn_metadata
    if a is None or a.num_prefills == 0: return None
    try:
        mapping = [(rid, sid) for rid, ids in (model_input.request_ids_to_seq_ids or {}).items() for sid in ids]
        if not any(rid in s['prompts'] for rid, _ in mapping):
            return None  # Initialization/profile work is not one of our four requests.
        # Matched default reference has exactly one full original prompt per prefill.
        old.require(a.num_prefills == 1 and len(mapping) == 1, 'one native full prefill only')
        rid, sid = mapping[0]
        old.require(rid in s['prompts'] and rid not in s['seen'] and len(s['seen']) < 4, 'explicit prefill UUID/cap')
        n = len(s['prompts'][rid]); p = a.prefill_metadata
        old.require(n in (96, 192) and a.num_prefill_tokens == n and a.num_decode_tokens == 0, 'original prefill shape')
        old.require(model_input.input_tokens.numel() == n and p.block_tables is not None and
                    p.block_tables.ndim == 2 and p.block_tables.shape[0] == 1 and p.block_tables.shape[1] <= 64,
                    'bounded actual prefill metadata')
        f = flags(runner, rank)
        old.require(rank in (0, 1) and all(f[k] is True for k in
            ('runner_chunked','builder_chunked','builder_config_chunked','runner_builder_config_same')) and
            f['tokens'] == 8192 and f['seqs'] == 32, 'actual all-True worker configuration')
        cache_numel = [x.numel() for x in kv_caches]
        old.require(len(cache_numel) == 64 and all(type(v) is int and v > 0 for v in cache_numel), '64 actual nonempty layer caches')
        fields = dict(input_tokens=model_input.input_tokens, positions=model_input.input_positions,
            slots=p.slot_mapping, block_tables=p.block_tables,
            seq_lens=p.seq_lens_tensor, query_start_loc=p.query_start_loc,
            context_lens=p.context_lens_tensor)
        old.require(all(v is not None and v.numel() <= 192 for v in fields.values()), 'metadata element cap')
        tensors = {k: v.detach().clone() for k, v in fields.items()}
        if s['writer'] is None: s['writer'] = Writer(s['root'], rank)
        old.require(s['writer'].rank == rank, 'rank stable')
        s['seen'].add(rid)
        return (s, dict(schema=1, spec_sha256=s['spec_sha'], request_id=rid, seq_id=sid,
            prompt_length=n, rank=rank, pid=os.getpid(), before_forward_s=time.time(), worker=f,
            num_prefills=a.num_prefills, num_prefill_tokens=a.num_prefill_tokens,
            num_decode_tokens=a.num_decode_tokens, block_tables_shape=list(p.block_tables.shape),
            block_tables_numel=p.block_tables.numel(), kv_cache_numel_per_layer=cache_numel,
            inferred_flash_prefill_branch='paged_kv' if p.block_tables.numel() else 'direct_kv',
            branch_is_source_inference=True, flash_source_sha256=FLASH_SHA), tensors)
    except Exception as exc:
        if s['writer'] is not None: s['writer'].failed=True; s['writer'].error=repr(exc)[:512]
        s['error'] = repr(exc)[:512]
        return None


def finish(context):
    if context is None: return
    s, record, tensors = context
    try:
        # CUDA synchronization is deliberately after actual sampling; affects later timing.
        record['metadata'] = {k: v.cpu().tolist() for k, v in tensors.items()}
        record['after_sampler_read_s'] = time.time()
        old.require(record['metadata']['input_tokens'] == s['prompts'][record['request_id']], 'actual original input tokens')
        s['writer'].submit(record)
    except Exception as exc:
        s['writer'].failed=True; s['writer'].error=repr(exc)[:512]
