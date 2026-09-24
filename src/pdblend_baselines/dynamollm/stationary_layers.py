"""Dynamo-only execution directly from original and incoming weight fragments.

Only activation tensors are assembled. No dense target weight, retained clone,
checkpoint loader or host weight staging is used here. CUDA arithmetic and TP
handoff remain unqualified until independent hardware/output acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from .stationary_ipc import StationaryConsumer, TorchCudaIpcCodec, digest, need
from .stationary_tensors import _tensor_identity


class BorrowableStationaryConsumer(StationaryConsumer):
    """Imported storage cannot be acknowledged released while layers borrow it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._borrowers = set()

    def borrow(self, borrower):
        self.receipt()
        need(borrower not in self._borrowers, 'duplicate stationary storage borrow')
        self._borrowers.add(borrower)

    def return_borrow(self, borrower):
        need(borrower in self._borrowers, 'unknown stationary storage borrow')
        self._borrowers.remove(borrower)

    def close(self):
        need(not self._borrowers, 'target layers still borrow original CUDA storage')
        return super().close()


class FragmentParameter:
    """One exact parameter, represented by disjoint slices, never concatenated."""

    def __init__(self, name, pieces, tensors, *, cpu_oracle=False):
        need(pieces and len(pieces) == len(tensors), 'parameter fragments missing')
        self.name = name
        self.shape = tuple(pieces[0]['target_shape'])
        self.axis = pieces[0]['axis']
        self.rows = sorted(zip(deepcopy(pieces), tensors), key=lambda row: row[0]['target_offset'])
        self.cpu_oracle = cpu_oracle
        self.closed = False
        cursor = 0
        for piece, value in self.rows:
            need(piece['parameter'] == name and tuple(piece['target_shape']) == self.shape
                 and piece['axis'] == self.axis, 'parameter fragment identity differs')
            expected = list(self.shape)
            if self.axis is not None:
                expected[self.axis] = piece['length']
                need(piece['target_offset'] == cursor, 'parameter gap or overlap')
                cursor += piece['length']
            else:
                need(len(self.rows) == 1 and piece['target_offset'] == 0
                     and piece['length'] == math.prod(self.shape), 'replicated parameter differs')
            need(isinstance(value, torch.Tensor) and not value.is_meta
                 and list(value.shape) == expected and not value.requires_grad,
                 'actual fragment shape/storage/autograd differs')
            need((cpu_oracle and value.device.type == 'cpu' and value.dtype in (torch.float64, torch.bfloat16))
                 or (not cpu_oracle and value.is_cuda and value.dtype == torch.bfloat16),
                 'only CUDA BF16 or explicit CPU arithmetic oracle allowed')
            # Strided row slices are valid GEMM views. Arbitrary/transposed views
            # might cause an implicit weight materialization in dispatch.
            need(value.ndim in (1, 2) and value.stride(-1) == 1
                 and (value.ndim == 1 or value.stride(0) >= value.shape[1]),
                 'unsupported fragment stride; no implicit retained contiguous copy')
        need(self.axis is None or cursor == self.shape[self.axis], 'incomplete parameter coverage')
        need(len({(v.device, v.dtype) for _, v in self.rows}) == 1, 'fragment device/dtype differs')
        self.device, self.dtype = self.rows[0][1].device, self.rows[0][1].dtype
        self.before = [_tensor_identity(v) for _, v in self.rows]

    def check(self):
        need(not self.closed, 'fragment parameter is closed')
        need([_tensor_identity(v) for _, v in self.rows] == self.before,
             'fragment storage or tensor value version changed')

    def _input(self, x):
        self.check()
        need(x.device == self.device and x.dtype == self.dtype and not x.requires_grad,
             'activation device/dtype/autograd differs')

    def add_vector(self, output):
        self._input(output)
        need(len(self.shape) == 1 and output.shape[-1] == self.shape[0], 'bias shape differs')
        for piece, value in self.rows:
            output[..., piece['target_offset']:piece['target_offset'] + value.numel()].add_(value)
        return output

    def linear(self, x, bias=None):
        self._input(x)
        need(len(self.shape) == 2 and self.axis in (0, 1) and x.ndim >= 1
             and x.shape[-1] == self.shape[1], 'linear activation/partition shape differs')
        output = x.new_zeros((*x.shape[:-1], self.shape[0]))
        for piece, weight in self.rows:
            offset, length = piece['target_offset'], piece['length']
            if self.axis == 0:
                output[..., offset:offset + length] = F.linear(x, weight)
            else:
                output.add_(F.linear(x[..., offset:offset + length], weight))
        if bias is not None:
            bias.add_vector(output)
        return output

    def embedding(self, token_ids):
        self.check()
        need(len(self.shape) == 2 and self.axis == 0 and token_ids.device == self.device
             and token_ids.dtype in (torch.int32, torch.int64), 'embedding input/partition differs')
        # An actual embedding lookup must reject invalid local IDs; masking must
        # not silently turn an invalid ID into a valid zero embedding.
        valid = ((token_ids >= 0) & (token_ids < self.shape[0])).all()
        if self.cpu_oracle:
            need(bool(valid), 'embedding local token outside vocabulary')
        else:
            torch._assert_async(valid, 'embedding local token outside vocabulary')
        output = torch.zeros((*token_ids.shape, self.shape[1]), device=self.device, dtype=self.dtype)
        for piece, weight in self.rows:
            offset, length = piece['target_offset'], piece['length']
            mask = (token_ids >= offset) & (token_ids < offset + length)
            local = (token_ids - offset).clamp(0, length - 1)
            output.add_(F.embedding(local, weight) * mask.unsqueeze(-1))
        return output

    def replicated_view(self):
        self.check()
        need(self.axis is None and len(self.rows) == 1, 'only replicated parameters bind as one view')
        return self.rows[0][1]

    def close(self):
        self.rows.clear()
        self.closed = True


class FragmentInventory:
    """Complete rank inventory; actual IPC imports own all retained fragments.

    Missing fragments must already have a completed direct GPU transport
    receipt. This module performs no transport and does not qualify a receipt
    supplied by a new transport implementation; hardware qualification remains
    false. CPU oracle construction is deliberately a separate named entrypoint.
    """

    @classmethod
    def cpu_oracle(cls, plan, target_rank, fragments):
        return cls(plan, target_rank, fragments, consumers=(), cpu_oracle=True)

    @classmethod
    def from_ipc(cls, plan, target_rank, *, target_codec, consumers, missing):
        target_uuid = plan['target_gpu_uuids'][target_rank]
        need(isinstance(target_codec, TorchCudaIpcCodec) and target_codec.uuid == target_uuid,
             'target CUDA device must have a validated physical UUID')
        fragments = []
        need(len({id(c) for c in consumers}) == len(consumers), 'duplicate IPC consumer')
        for consumer in consumers:
            need(isinstance(consumer, BorrowableStationaryConsumer), 'borrow-protected actual IPC consumer required')
            consumer.receipt()
            need(consumer.packet['plan_sha256'] == plan['plan_sha256']
                 and consumer.packet['target_rank'] == target_rank and consumer.codec.uuid == target_uuid,
                 'IPC consumer belongs to another target plan/rank')
            fragments.extend(consumer.views)
        for row in missing:
            piece, value, receipt = row['piece'], row['tensor'], row['receipt']
            need(piece['kind'] == 'direct_gpu_transfer' and piece['source_gpu_uuid'] != target_uuid
                 and piece['target_gpu_uuid'] == target_uuid, 'only missing remote fragments may be transferred')
            need(isinstance(value, torch.Tensor) and value.is_cuda and value.dtype == torch.bfloat16
                 and value.is_contiguous() and value.storage_offset() == 0
                 and value.untyped_storage().nbytes() == piece['bytes'],
                 'incoming allocation must contain exactly one missing fragment, not a dense target weight')
            need(receipt.get('schema') == 'dynamo-missing-fragment-transport/v1'
                 and receipt.get('plan_sha256') == plan['plan_sha256']
                 and receipt.get('piece') == piece and receipt.get('completed') is True
                 and receipt.get('path') == 'direct_cuda_to_cuda'
                 and receipt.get('host_weight_staging_bytes') == 0
                 and receipt.get('retained_fragment_copy_bytes') == 0
                 and receipt.get('bytes') == piece['bytes']
                 and receipt.get('destination') == _tensor_identity(value),
                 'actual completed missing-fragment transport receipt required')
            fragments.append((piece, value))
        result = cls(plan, target_rank, fragments, consumers=consumers, cpu_oracle=False)
        try:
            need(all(p.device == torch.device('cuda', target_codec.device) for p in result.parameters.values()),
                 'target fragments do not use the validated physical CUDA device')
        except BaseException:
            result.close()
            raise
        result.missing_receipts = [deepcopy(row['receipt']) for row in missing]
        return result

    def __init__(self, plan, target_rank, fragments, *, consumers, cpu_oracle):
        need(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}), 'tensor plan changed')
        need(type(target_rank) is int and 0 <= target_rank < len(plan['target_gpu_uuids']), 'invalid target rank')
        expected = [p for p in plan['pieces'] if p['target_rank'] == target_rank]
        need(sorted(digest(p) for p, _ in fragments) == sorted(digest(p) for p in expected),
             'exact target inventory has duplicate, missing or foreign fragments')
        self.plan, self.target_rank = deepcopy(plan), target_rank
        self.is_cpu_oracle, self.closed = cpu_oracle, False
        self.consumers, self.parameters, self.missing_receipts = list(consumers), {}, []
        self._bindings = set()
        borrowed = []
        try:
            for consumer in self.consumers:
                consumer.borrow(self)
                borrowed.append(consumer)
            for name in plan['target_shapes']:
                selected = [(p, v) for p, v in fragments if p['parameter'] == name]
                self.parameters[name] = FragmentParameter(name, [p for p, _ in selected],
                    [v for _, v in selected], cpu_oracle=cpu_oracle)
            devices = {p.device for p in self.parameters.values()}
            need(len(devices) == 1, 'target rank fragments cross devices')
            if not cpu_oracle:
                for consumer in self.consumers:
                    need(next(iter(devices)) == torch.device('cuda', consumer.codec.device),
                         'target fragments do not use the validated IPC CUDA device')
        except BaseException:
            for p in self.parameters.values():
                p.close()
            for consumer in borrowed:
                consumer.return_borrow(self)
            raise

    def check(self):
        need(not self.closed, 'fragment inventory is closed')
        for consumer in self.consumers:
            consumer.receipt()
        for p in self.parameters.values():
            p.check()

    def borrow_binding(self, binding):
        self.check()
        need(binding not in self._bindings, 'duplicate model binding')
        self._bindings.add(binding)

    def return_binding(self, binding):
        need(binding in self._bindings, 'unknown model binding')
        self._bindings.remove(binding)

    def receipt(self):
        self.check()
        return dict(schema='dynamo-segmented-parameter-inventory/v1', plan_sha256=self.plan['plan_sha256'],
            target_rank=self.target_rank, target_gpu_uuid=self.plan['target_gpu_uuids'][self.target_rank],
            parameters={name: [dict(piece=p, tensor=_tensor_identity(v)) for p, v in param.rows]
                        for name, param in self.parameters.items()}, cpu_oracle=self.is_cpu_oracle,
            retained_fragment_copy_bytes=0, dense_target_weight_allocated_bytes=0, host_weight_staging_bytes=0,
            missing_transport_receipts=self.missing_receipts, hardware_qualified=False,
            target_engine_activated=False, formal_eligible=False)

    def close(self):
        need(not self._bindings, 'model still bound to fragment inventory')
        need(not self.closed, 'fragment inventory already closed')
        for p in self.parameters.values():
            p.close()
        for consumer in self.consumers:
            consumer.return_borrow(self)
        self.closed = True


class SegmentedQuantMethod:
    """Per-model vLLM apply/embedding interface; TP collectives stay in vLLM."""

    def __init__(self, inventory, weight_name, bias_name=None):
        self.inventory, self.weight_name, self.bias_name = inventory, weight_name, bias_name
        self.closed = False

    def _weight(self):
        need(not self.closed, 'segmented target layer is closed')
        # Check owner lifetime as well as the particular weight on every call.
        need(not self.inventory.closed, 'fragment inventory is closed')
        for consumer in self.inventory.consumers:
            consumer.receipt()
        return self.inventory.parameters[self.weight_name]

    def apply(self, layer, x, bias=None):
        need(bias is None, 'dense/external bias bypasses exact fragment binding')
        return self._weight().linear(x, None if self.bias_name is None else self.inventory.parameters[self.bias_name])

    def embedding(self, layer, input_):
        return self._weight().embedding(input_)

    def create_weights(self, *args, **kwargs):
        raise RuntimeError('stationary target must bind fragments; dense target allocation prohibited')

    def process_weights_after_loading(self, *args, **kwargs):
        raise RuntimeError('stationary target does not use a dense checkpoint/dummy weight loader')
