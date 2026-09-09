"""Request-addressed V0 NCCL transfer for Qwen-style GQA, with TP reslicing.

Both capabilities are initialized once. Mixed requests do not send or receive.
P and D can have different tensor-parallel degrees; peers may change roles
only after the controller drains their requests and outstanding transfers.
Pipeline parallelism is deliberately rejected until layer ownership is tested.
"""
import torch
import hashlib
import json
import os
import time
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import model_aware_kv_ops_helper
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import P2pNcclEngine
from vllm.pdblend_runtime import parse_transfer


class PDBlendConnector(KVConnectorBase):
    def __init__(self, rank, local_rank, config):
        if config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("PDBlendConnector PP>1 is not validated")
        self.config = config.kv_transfer_config
        self.tp = config.parallel_config.tensor_parallel_size
        self.rank = rank
        self.helper = model_aware_kv_ops_helper(config)
        self.peers = self.config.kv_connector_extra_config["peers"]
        self.engine_id = self.config.engine_id
        self.validation = []
        self.verify_transport=self.config.get_from_extra_config('verify_transport',False)
        self.transport = P2pNcclEngine(local_rank, self.config,
                                      hostname=self.config.kv_ip, port_offset=rank)

    def _requests(self, model_input, phase):
        ids = list(model_input.request_ids_to_seq_ids or {})
        lengths = model_input.attn_metadata.seq_lens
        offset = 0
        prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        for request_id, length in zip(ids, lengths):
            if offset >= prefill_tokens:
                break
            meta = parse_transfer(request_id)
            if not meta or meta["phase"] != phase:
                raise ValueError("mixed transfer/local prefill batch is unsupported")
            if offset + length > prefill_tokens:
                raise ValueError("transfer requires complete, unchunked prefill")
            yield meta, offset, offset + length
            offset += length
        if phase == "d" and offset != len(model_input.input_tokens):
            raise ValueError("decode import must be scheduled separately from running decode")

    def send_kv_caches_and_hidden_states(self, model_executable, model_input,
                                        kv_caches, hidden_or_intermediate_states):
        torch.cuda.current_stream().synchronize()
        transfer_started = time.time()
        num_heads, head_size = self.helper.get_model_args(model_executable)
        total_heads = num_heads * self.tp
        lo, hi = self.rank * num_heads, (self.rank + 1) * num_heads
        slots = model_input.attn_metadata.slot_mapping.flatten()
        for meta, start, end in self._requests(model_input, "p"):
            if meta["source"] != self.engine_id:
                raise ValueError("wrong KV producer")
            peer = self.peers[meta["target"]]
            dest_tp = int(peer["tp"])
            if total_heads % dest_tp:
                raise ValueError("KV heads not divisible by destination TP")
            layers = [self.helper.get_kv_from_cache(kv, num_heads, head_size)
                      for kv in kv_caches]
            keys = torch.stack([k[slots[start:end]] for k, _ in layers])
            values = torch.stack([v[slots[start:end]] for _, v in layers])
            for target_rank in range(dest_tp):
                a, b = target_rank * total_heads // dest_tp, (target_rank + 1) * total_heads // dest_tp
                left, right = max(a, lo), min(b, hi)
                address = "%s:%s" % (peer["host"], int(peer["kv_port"]) + target_rank)
                stem = "%s#%d" % (meta["nonce"], self.rank)
                if left < right:
                    for name, tensor in (("k", keys), ("v", values)):
                        part = tensor[:, :, left-lo:right-lo].contiguous()
                        self.record_tensor('export',stem+name,part,target_rank=target_rank)
                        if not self.transport.send_tensor(stem + name, part, address):
                            raise RuntimeError("NCCL receiver refused KV")
                if self.rank == 0:
                    self.record_tensor('export',meta['nonce']+'#hidden',
                        hidden_or_intermediate_states[start:end],target_rank=target_rank)
                    if not self.transport.send_tensor(meta["nonce"] + "#hidden",
                            hidden_or_intermediate_states[start:end].contiguous(), address):
                        raise RuntimeError("NCCL receiver refused hidden states")
        self.record_transfer("send", model_input, transfer_started)

    def recv_kv_caches_and_hidden_states(self, model_executable, model_input, kv_caches):
        transfer_started = time.time()
        num_heads, head_size = self.helper.get_model_args(model_executable)
        total_heads = num_heads * self.tp
        lo, hi = self.rank * num_heads, (self.rank + 1) * num_heads
        slots = model_input.attn_metadata.slot_mapping.flatten()
        hidden = []
        for meta, start, end in self._requests(model_input, "d"):
            if meta["target"] != self.engine_id:
                raise ValueError("wrong KV consumer")
            source_tp = int(self.peers[meta["source"]]["tp"])
            if total_heads % source_tp:
                raise ValueError("KV heads not divisible by source TP")
            parts_k, parts_v = [], []
            for source_rank in range(source_tp):
                a, b = source_rank * total_heads // source_tp, (source_rank + 1) * total_heads // source_tp
                if max(a, lo) >= min(b, hi):
                    continue
                stem = "%s#%d" % (meta["nonce"], source_rank)
                key=self.transport.recv_tensor(stem+'k')
                value=self.transport.recv_tensor(stem+'v')
                self.record_tensor('import',stem+'k',key,target_rank=self.rank)
                self.record_tensor('import',stem+'v',value,target_rank=self.rank)
                parts_k.append(key)
                parts_v.append(value)
            # Equal/finer target TP receives one already-contiguous head
            # slice. Reuse it instead of allocating another full KV copy.
            keys = parts_k[0] if len(parts_k)==1 else torch.cat(parts_k,dim=2)
            values = parts_v[0] if len(parts_v)==1 else torch.cat(parts_v,dim=2)
            if keys.shape[1:3] != (end-start, num_heads):
                raise ValueError("KV shape mismatch")
            if self.config.get_from_extra_config("verify_recompute", False):
                self.validation.append((meta, start, end, keys, values))
            for layer_id, cache in enumerate(kv_caches):
                layer = model_executable.model.layers[layer_id]
                self.helper.put_kv_to_cache(model_executable, keys[layer_id],
                    values[layer_id], layer, cache, slots, start, end)
                if self.verify_transport:
                    local_k,local_v=self.helper.get_kv_from_cache(cache,num_heads,head_size)
                    if not (torch.equal(local_k[slots[start:end]],keys[layer_id]) and
                            torch.equal(local_v[slots[start:end]],values[layer_id])):
                        raise RuntimeError('imported KV differs from destination cache readback')
            if self.verify_transport:
                self.record_validation(dict(direction='cache_readback',nonce=meta['nonce'],
                    target_rank=self.rank,layers=len(kv_caches),bit_exact=True))
            h=self.transport.recv_tensor(meta['nonce']+'#hidden')
            self.record_tensor('import',meta['nonce']+'#hidden',h,target_rank=self.rank)
            hidden.append(h)
        result = hidden[0] if len(hidden)==1 else torch.cat(hidden)
        torch.cuda.current_stream().synchronize()
        self.record_transfer("receive", model_input, transfer_started)
        return result, not bool(self.validation), model_input

    def validate_recomputed(self, model_executable, model_input, kv_caches):
        """Diagnostic only: recompute D prefill and compare numerical KV.

        Different TP reductions need not be bit-identical in BF16. The fixed
        5% relative RMS bound detects head/order corruption and is recorded;
        diagnostic recomputation is never eligible for performance evidence.
        """
        if not self.validation:
            return
        num_heads, head_size = self.helper.get_model_args(model_executable)
        slots = model_input.attn_metadata.slot_mapping.flatten()
        try:
            for meta, start, end, keys, values in self.validation:
                residual = denominator = 0.
                layers = []
                for layer_id, cache in enumerate(kv_caches):
                    local_k, local_v = self.helper.get_kv_from_cache(cache, num_heads, head_size)
                    error = norm = 0.
                    for received, local in ((keys[layer_id],local_k),(values[layer_id],local_v)):
                        expected=local[slots[start:end]].float()
                        error += (received.float()-expected).square().sum().item()
                        norm += expected.square().sum().item()
                    residual += error
                    denominator += norm
                    layers.append((error/max(norm,1e-20))**.5)
                nrmse=(residual/max(denominator,1e-20))**.5
                event=dict(request=meta,rank=self.rank,relative_rms=nrmse,
                           per_layer_relative_rms=layers,threshold=.05,passed=nrmse<=.05,
                           purpose="numerical_correctness_only")
                path=os.environ["PDBLEND_RUNTIME_PATH"]+".validation.%d.jsonl"%self.rank
                with open(path,"a") as handle:
                    handle.write(json.dumps(event)+"\n")
                if nrmse>.05:
                    raise RuntimeError("transferred KV differs from recomputed KV: %f"%nrmse)
        finally:
            self.validation.clear()

    def record_transfer(self, direction, model_input, started):
        path = os.environ.get("PDBLEND_RUNTIME_PATH")
        if path:
            event = dict(direction=direction, engine_id=self.engine_id, rank=self.rank,
                         request_ids=list(model_input.request_ids_to_seq_ids or {}),
                         started_s=started, finished_s=time.time())
            with open(path + ".kv.%d.jsonl" % self.rank, "a") as handle:
                handle.write(json.dumps(event) + "\n")

    def record_tensor(self,direction,tensor_id,tensor,*,target_rank):
        if self.verify_transport:
            payload=tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            self.record_validation(dict(direction=direction,tensor_id=tensor_id,
                target_rank=target_rank,shape=list(tensor.shape),dtype=str(tensor.dtype),
                sha256=hashlib.sha256(payload).hexdigest()))

    def record_validation(self,event):
        path=os.environ['PDBLEND_RUNTIME_PATH']+'.transport-validation.%d.jsonl'%self.rank
        with open(path,'a') as handle:
            handle.write(json.dumps(dict(event,engine_id=self.engine_id,source_rank=self.rank,
                purpose='bit-exact transport correctness; not performance'))+'\n')

    def close(self):
        self.transport.close()
