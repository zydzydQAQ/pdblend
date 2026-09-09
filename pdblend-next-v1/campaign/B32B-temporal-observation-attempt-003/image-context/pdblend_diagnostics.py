"""Opt-in, bounded observation only. Never receives a KV tensor or changes outputs."""
import atexit
import hashlib
import json
import os
import pathlib
import queue
import re
import threading
import time

IMAGE = 'sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
PARENT = '87ba22eba9b39316e990877b6855f79e78602ad09452d112ad3f78a29960327f'
STEPS = tuple(range(28, 35))
MAX_RECORDS = 14
MAX_BYTES = 16384
MAX_BLOCKS = 64
_STATE = None
_LOADED = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_spec(s):
    require(s['schema'] == 1 and s['enabled'] is True, 'explicit schema/enabled')
    require(s['parent_image'] == IMAGE and s['parent_model_runner_sha256'] == PARENT,
            'actual source/image binding')
    ids = s['request_ids']
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == 2 and
            all(isinstance(x, str) and re.fullmatch('[0-9a-f]{32}', x) for x in ids),
            'exactly two explicit UUID hex request IDs')
    require(s['output_indices'] == list(STEPS), 'fixed seven output positions')
    require(s['tp'] == 2 and s['pp'] == 1 and s['max_output_tokens'] == 64,
            'TP2 PP1 full64 only')
    p = pathlib.Path(s['output_dir'])
    base = pathlib.Path('/root/workspace/pdblend-next-v1/campaign')
    require(p.is_absolute() and base in p.parents and '..' not in p.parts,
            'independent absolute campaign output')
    require(s['diagnostic_only'] is True and s['performance_claims_allowed'] is False,
            'observation is not performance evidence')
    return s


class Writer:
    """At most 32 CPU-only strings. File I/O never runs on the model thread."""
    def __init__(self, root, rank):
        self.root = pathlib.Path(root)
        self.rank = rank
        self.pid = os.getpid()
        self.q = queue.Queue(maxsize=32)
        self.closed = threading.Event()
        self.failed = False
        self.written = 0
        self.submitted = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='pdblend-diagnostic-writer')
        self.thread.start()
        atexit.register(self.close)

    def submit(self, value):
        try:
            payload = json.dumps(value, sort_keys=True, allow_nan=False) + '\n'
            require(len(payload.encode()) <= MAX_BYTES, 'record byte cap')
            require(self.submitted < MAX_RECORDS, 'record count cap')
            self.q.put_nowait(payload)
            self.submitted += 1
        except Exception as exc:
            self.failed = True
            self.error = repr(exc)[:512]

    def _status(self):
        p = self.root / ('rank%d-pid%d.status.json' % (self.rank, self.pid))
        value = dict(schema=1, rank=self.rank, pid=self.pid, failed=self.failed,
                     error=self.error, written=self.written,
                     complete=(self.written == MAX_RECORDS and not self.failed),
                     expected_records=MAX_RECORDS, time_s=time.time())
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, sort_keys=True) + '\n')
        tmp.replace(p)

    def _run(self):
        try:
            # The deployment must create a new empty directory; no existing log overwrite.
            self.root.mkdir(parents=True, exist_ok=True)
            p = self.root / ('rank%d-pid%d.jsonl' % (self.rank, self.pid))
            with p.open('x') as f:
                self._status()
                while not self.closed.is_set() or not self.q.empty():
                    try:
                        payload = self.q.get(timeout=.1)
                    except queue.Empty:
                        if self.failed and not getattr(self, '_failure_persisted', False):
                            self._status()
                            self._failure_persisted = True
                        continue
                    f.write(payload)
                    f.flush()
                    self.written += 1
                    self._status()
                    # Remain alive until process cleanup so late error/duplicate is visible.
                self._status()  # Preserve a final error even when close races the idle loop.
        except Exception as exc:
            self.failed = True
            self.error = repr(exc)[:512]
            # A missing/error status is never accepted by the offline validator.
            try:
                self._status()
            except Exception:
                pass

    def close(self):
        self.closed.set()
        self.thread.join(timeout=.25)


class State:
    def __init__(self, spec, spec_sha):
        self.spec = validate_spec(spec)
        self.spec_sha = spec_sha
        self.ids = frozenset(spec['request_ids'])
        self.seen = set()
        self.writer = None
        self.error = None

    def fail(self, exc):
        if self.error is None:
            self.error = repr(exc)[:512]
        if self.writer is not None:
            self.writer.failed = True
            self.writer.error = self.error

    def writer_for(self, rank):
        require(rank in (0, 1), 'TP rank must be 0/1')
        if self.writer is None:
            self.writer = Writer(self.spec['output_dir'], rank)
        require(self.writer.rank == rank, 'rank changed within process')
        return self.writer


def state():
    global _STATE, _LOADED
    if not _LOADED:
        _LOADED = True
        p = os.environ.get('PDBLEND_DIAGNOSTIC_SPEC')
        if p:
            try:
                raw = pathlib.Path(p).read_bytes()
                require(len(raw) <= 16384, 'spec byte cap')
                digest = hashlib.sha256(raw).hexdigest()
                require(digest == os.environ.get('PDBLEND_DIAGNOSTIC_SPEC_SHA256'),
                        'spec SHA mismatch')
                _STATE = State(json.loads(raw), digest)
            except Exception as exc:
                # One bounded warning, no inference exception or untrusted path write.
                try:
                    os.write(2, ('PDB diagnostic disabled: %r\n' % exc)[:768].encode())
                except OSError:
                    pass
    return _STATE


def bind(groups, model_input, sampling_metadata, block_size):
    """Driver CPU UUID->seq->input row and independent sampler row binding."""
    s = state()
    if s is None or s.error:
        return None
    try:
        selected = [g for g in groups if g.request_id in s.ids]
        if not selected:
            return None
        result = []
        mapping = model_input.request_ids_to_seq_ids
        ordered = [(rid, sid) for rid, ids in mapping.items() for sid in ids]
        require(len(ordered) == len(model_input.query_lens) <= 32, 'input mapping shape')
        require(all(isinstance(rid, str) and len(rid) <= 128 and type(sid) is int
                    for rid, sid in ordered), 'bounded identifier metadata')
        require(len(set(sid for _, sid in ordered)) == len(ordered), 'unique sequence IDs')
        for group in selected:
            if group.is_prompt or not group.do_sample:
                continue
            require(len(group.seq_data) == 1, 'single-sequence sampling only')
            sid, data = next(iter(group.seq_data.items()))
            output_index = data.get_output_len() + 1
            if output_index not in STEPS:
                continue
            require(mapping[group.request_id] == [sid], 'UUID/seq mapping')
            row = ordered.index((group.request_id, sid))
            require(model_input.query_lens[row] == 1, 'one decode query token')
            prompt_len = data.get_prompt_len()
            require(prompt_len in (96, 192), 'original prompt lengths')
            params = group.sampling_params
            require(params.max_tokens == 64 and params.temperature == 0 and
                    params.top_p == 1 and params.ignore_eos is True and params.n == 1 and params.seed == 0,
                    'original greedy full64 sampling')
            sample_groups = [g for g in sampling_metadata.seq_groups if sid in g.seq_ids]
            require(len(sample_groups) == 1 and sample_groups[0].seq_ids == [sid] and
                    len(sample_groups[0].sample_indices) == 1, 'explicit logits row mapping')
            table = list(group.block_tables[sid])
            require(0 < len(table) <= MAX_BLOCKS and type(block_size) is int and block_size > 0,
                    'bounded physical block table')
            result.append(dict(request_id=group.request_id, seq_id=sid,
                output_index=output_index, input_sequence_row=row,
                input_token_offset=sum(model_input.query_lens[:row]),
                logits_row=sample_groups[0].sample_indices[0],
                prompt_len=prompt_len, output_len_before=data.get_output_len(),
                computed_before=data.get_num_computed_tokens(), sequence_len=data.get_len(),
                last_input_token=data.get_last_token_id(), driver_block_table=table,
                block_size=block_size, expected_query_len=1,
                ordered_request_seq_ids=ordered))
        return dict(schema=1, spec_sha256=s.spec_sha, records=result) if result else None
    except Exception as exc:
        s.fail(exc)
        return None


def begin(model_input, rank, tp, pp, eager):
    """Snapshot only small metadata tensors before forward; no host CUDA read yet."""
    s = state()
    if s is None or s.error:
        return None
    try:
        s.writer_for(rank)
        b = model_input.diagnostic_bindings
        require(tp == 2 and pp == 1 and eager is True, 'actual TP2 eager/PP1 gate')
        require(b['spec_sha256'] == s.spec_sha and 1 <= len(b['records']) <= 2,
                'broadcast observation spec binding')
        a = model_input.attn_metadata
        require(not a.use_cuda_graph, 'eager metadata only')
        import torch
        contexts = []
        enqueue_started = time.perf_counter()
        for record in b['records']:
            d = dict(record)
            rid, step, sid = d['request_id'], d['output_index'], d['seq_id']
            require(rid in s.ids and step in STEPS, 'explicit allowed UUID/step')
            require((rid, step) not in s.seen and len(s.seen) < MAX_RECORDS, 'duplicate/cap')
            mapping = [(r, seq) for r, ids in model_input.request_ids_to_seq_ids.items() for seq in ids]
            require(mapping == [tuple(x) for x in d['ordered_request_seq_ids']], 'rank mapping parity')
            row = mapping.index((rid, sid))
            require(row == d['input_sequence_row'], 'bound UUID input row')
            off = d['input_token_offset']
            require(a.block_tables.ndim == 2 and 0 < a.block_tables.shape[1] <= MAX_BLOCKS,
                    'actual block-table read cap')
            require(0 <= off < model_input.input_tokens.numel() and model_input.input_positions.ndim == 1,
                    'bounded token offset')
            fields = {
                'input_token': model_input.input_tokens[off:off+1],
                'position': model_input.input_positions[off:off+1],
                'slot': a.slot_mapping[off:off+1],
                'sequence_len': a.seq_lens_tensor[row:row+1],
                'context_len': a.context_lens_tensor[row:row+1],
                'query_start_end': a.query_start_loc[row:row+2],
                'block_table': a.block_tables[row],
            }
            lengths = {k: v.numel() for k, v in fields.items()}
            require(all(lengths[k] == 1 for k in ('input_token','position','slot','sequence_len','context_len'))
                    and lengths['query_start_end'] == 2, 'actual metadata dimensions')
            packed = torch.cat([v.detach().reshape(-1).to(dtype=torch.int64) for v in fields.values()]).clone()
            s.seen.add((rid, step))
            d.update(rank=rank, pid=os.getpid(), spec_sha256=s.spec_sha,
                     snapshot_enqueued_s=time.time(), tp=tp, pp=pp, eager=eager,
                     tensor_device=str(model_input.input_tokens.device),
                     cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                     num_prefills=a.num_prefills, num_prefill_tokens=a.num_prefill_tokens,
                     num_decode_tokens=a.num_decode_tokens)
            contexts.append(dict(record=d, metadata=packed, lengths=lengths, raw_logits=None))
        for c in contexts:
            c['record']['metadata_enqueue_host_s'] = time.perf_counter() - enqueue_started
        return contexts
    except Exception as exc:
        s.fail(exc)
        return None


def capture_logits(contexts, logits):
    """Read-only GPU top2+two values before sampler mutates its logits input."""
    if not contexts:
        return
    s = state()
    try:
        import torch
        for c in contexts:
            started = time.perf_counter()
            require(c['record']['rank'] == 0 and logits is not None and logits.ndim == 2,
                    'rank0 actual model logits')
            row = c['record']['logits_row']
            require(0 <= row < logits.shape[0] and logits.shape[1] > 4172, 'bound logits row')
            values, indices = torch.topk(logits[row], 2)
            # Only 7 scalar values cross to CPU. Top-k scans the selected vocab row on GPU.
            c['raw_logits'] = torch.cat((values.to(torch.float64), indices.to(torch.float64),
                logits[row, 2776:2777].to(torch.float64), logits[row, 4172:4173].to(torch.float64),
                torch.argmax(logits[row]).reshape(1).to(torch.float64))).clone()
            c['record']['raw_logits_dtype'] = str(logits.dtype)
            c['record']['raw_logits_shape'] = list(logits.shape)
            c['record']['logits_enqueue_host_s'] = time.perf_counter() - started
    except Exception as exc:
        s.fail(exc)


def finish(contexts, output):
    """Small CUDA D2H read synchronizes selected stream work; record this disturbance."""
    if not contexts:
        return
    s = state()
    try:
        for c in contexts:
            d = c['record']
            started = time.perf_counter()
            # No KV contents and no full-vocabulary CPU transfer.
            flat = c['metadata'].cpu().tolist()
            actual, offset = {}, 0
            for name, size in c['lengths'].items():
                actual[name] = flat[offset:offset+size]
                offset += size
            d['actual'] = actual
            if d['rank'] == 0:
                require(c['raw_logits'] is not None and output is not None, 'logits/sampler evidence')
                vals = c['raw_logits'].cpu().tolist()
                samples = [v for g in output.outputs for v in g.samples
                           if v.parent_seq_id == d['seq_id']]
                require(len(samples) == 1, 'sampler output explicit parent_seq_id')
                d['raw_model_logits_pre_sampler'] = dict(top2_values=vals[:2],
                    top2_token_ids=[int(x) for x in vals[2:4]], token_2776=vals[4],
                    token_4172=vals[5], argmax_token_id=int(vals[6]))
                d['sampler_output_token'] = samples[0].output_token
                d['sampler_parent_seq_id'] = samples[0].parent_seq_id
            else:
                require(output is None, 'non-driver has no invented sampler evidence')
                d['raw_model_logits_pre_sampler'] = None
                d['sampler_output_token'] = None
            d['host_readback_s'] = time.perf_counter() - started
            d['recorded_s'] = time.time()
            d['observer_error'] = s.error
            d['scope'] = 'metadata-and-pre-sampler-logits; no KV-content proof'
            s.writer.submit(d)
    except Exception as exc:
        s.fail(exc)
