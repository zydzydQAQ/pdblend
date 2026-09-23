"""DistServe stage queues, ported from upstream 82831f1 (Apache-2.0).

Modified: engine-independent request records, explicit completion/transfer ACKs,
and no Ray/Torch dependencies. See baselines/distserve/references/manifest.json.
This module makes decisions; it does not claim an engine executed a batch.
"""
from collections import deque
from dataclasses import dataclass, field


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass
class Request:
    request_id: str
    input_tokens: int
    output_tokens: int
    payload: dict
    generated: int = 0
    source: str | None = None
    target: str | None = None
    target_virtual: int | None = None
    kv_handle: str | None = None
    cancelled: bool = False
    queue: object = field(default=None, repr=False)

    def __post_init__(self):
        positive_int(self.input_tokens, 'input_tokens')
        positive_int(self.output_tokens, 'output_tokens')

    def blocks(self, block_size=16, *, prompt_only=False):
        return (self.input_tokens + (0 if prompt_only else self.generated) + block_size - 1)//block_size

    @property
    def context_tokens(self):
        return self.input_tokens + self.generated


class PrefillScheduler:
    def __init__(self, *, max_batch_size, max_tokens_per_batch, num_gpu_blocks, block_size=16):
        self.max_batch_size=positive_int(max_batch_size, 'max_batch_size')
        self.max_tokens_per_batch=positive_int(max_tokens_per_batch, 'max_tokens_per_batch')
        self.num_gpu_blocks=positive_int(num_gpu_blocks, 'num_gpu_blocks')
        self.block_size=positive_int(block_size, 'block_size')
        self.waiting=deque()
        self.processing={}
        self.retained={}

    def add(self, request):
        if any(r.request_id==request.request_id for r in self.waiting) or request.request_id in self.processing or request.request_id in self.retained:
            raise ValueError('duplicate request')
        self.waiting.append(request)

    @property
    def retained_blocks(self):
        return sum(r.blocks(self.block_size,prompt_only=True) for r in self.retained.values())

    def next_batch(self, *, free_gpu_blocks=None):
        # Author context scheduler includes the KV bridge and in-flight batches.
        occupied=self.retained_blocks+sum(r.blocks(self.block_size,prompt_only=True) for r in self.processing.values())
        available=max(0,self.num_gpu_blocks-occupied)
        if free_gpu_blocks is not None:
            available=min(available,max(0,free_gpu_blocks))
        batch=[];tokens=0;blocks=0
        while self.waiting and len(batch)<self.max_batch_size:
            request=self.waiting[0]
            if request.cancelled:
                self.waiting.popleft();continue
            needed=request.blocks(self.block_size,prompt_only=True)
            if tokens+request.input_tokens>self.max_tokens_per_batch or blocks+needed>available:
                break  # FCFS: never inspect younger requests for a fit.
            self.waiting.popleft();batch.append(request)
            tokens+=request.input_tokens;blocks+=needed
            self.processing[request.request_id]=request
        return tuple(batch)

    def complete(self, batch):
        for request in batch:
            self.processing.pop(request.request_id,None)
            request.generated=max(request.generated,1)
            self.retained[request.request_id]=request

    def release(self, request_id):
        self.retained.pop(request_id,None)

    def cancel(self, request_id):
        for request in tuple(self.waiting):
            if request.request_id==request_id:
                request.cancelled=True;self.waiting.remove(request);return
        request=self.processing.get(request_id) or self.retained.get(request_id)
        if request is not None: request.cancelled=True


class DecodeScheduler:
    def __init__(self, *, max_batch_size, max_tokens_per_batch, num_gpu_blocks,
                 block_size=16, waiting_block_prop_threshold=.05):
        self.max_batch_size=positive_int(max_batch_size, 'max_batch_size')
        self.max_tokens_per_batch=positive_int(max_tokens_per_batch, 'max_tokens_per_batch')
        self.num_gpu_blocks=positive_int(num_gpu_blocks, 'num_gpu_blocks')
        self.block_size=positive_int(block_size, 'block_size')
        if not 0<waiting_block_prop_threshold<=1: raise ValueError('invalid waiting block threshold')
        self.waiting_block_prop_threshold=waiting_block_prop_threshold
        self.bridge=deque();self.waiting=deque();self.active={}

    @property
    def load(self):
        # Real official scheduler counts waiting + processing, not unaccepted.
        return len(self.waiting)+len(self.active)

    def add_bridge(self, request):
        self.bridge.append(request)

    def accept_next(self, *, free_gpu_blocks):
        while self.bridge and self.bridge[0].cancelled:self.bridge.popleft()
        if not self.bridge:return None
        request=self.bridge[0]
        waiting_blocks=sum(r.blocks(self.block_size,prompt_only=True) for r in self.waiting)
        if (waiting_blocks>=self.num_gpu_blocks*self.waiting_block_prop_threshold
                or request.blocks(self.block_size,prompt_only=True)>free_gpu_blocks):
            return None
        self.bridge.popleft();self.waiting.append(request)
        return request

    def next_batch(self):
        tokens=sum(r.context_tokens for r in self.active.values())
        # Author checks waiting prompt blocks and current decode blocks together.
        occupied=sum(r.blocks(self.block_size) for r in self.active.values())
        occupied+=sum(r.blocks(self.block_size,prompt_only=True) for r in self.waiting)
        while self.waiting and len(self.active)<self.max_batch_size:
            request=self.waiting[0]
            if request.cancelled:
                self.waiting.popleft();continue
            extra=request.blocks(self.block_size)-request.blocks(self.block_size,prompt_only=True)
            if tokens+request.context_tokens>self.max_tokens_per_batch or occupied+extra>self.num_gpu_blocks:
                break
            self.waiting.popleft();self.active[request.request_id]=request
            tokens+=request.context_tokens;occupied+=extra
        return tuple(self.active.values())

    def finish(self, request_id):
        self.active.pop(request_id,None)
        for queue in (self.waiting,self.bridge):
            for request in tuple(queue):
                if request.request_id==request_id:queue.remove(request)


class PipelineDecodeScheduler(DecodeScheduler):
    """Author PP disjoint batch queues with explicit fixed-engine KV arenas.

    The official decoder rotates #PP batch queues; request/token constraints
    belong to each queue, while blocks count all queues and accepted waiting.
    The shared vLLM engine additionally partitions KV capacity by virtual arena.
    Migration therefore reserves an arena before pull; this fixed-engine
    constraint is recorded and is not the author's fungible block allocator.
    """
    def __init__(self,*,pipeline_parallel_size,**kwargs):
        super().__init__(**kwargs)
        self.pipeline_parallel_size=positive_int(pipeline_parallel_size,'pipeline_parallel_size')
        if self.num_gpu_blocks%self.pipeline_parallel_size:raise ValueError('equal virtual block arenas required')
        self.batch_queues=[{} for _ in range(self.pipeline_parallel_size)];self.cur_index=0

    def _virtual_requests(self,virtual):
        return list(self.batch_queues[virtual].values())+[r for r in self.waiting if r.target_virtual==virtual]

    def accept_next(self,*,free_gpu_blocks,free_blocks_by_virtual):
        while self.bridge and self.bridge[0].cancelled:self.bridge.popleft()
        if not self.bridge:return None
        if len(free_blocks_by_virtual)!=self.pipeline_parallel_size:raise ValueError('all virtual capacity receipts required')
        request=self.bridge[0]
        waiting_blocks=sum(r.blocks(self.block_size,prompt_only=True) for r in self.waiting)
        occupied=sum(r.blocks(self.block_size) for r in self.active.values())
        occupied+=sum(r.blocks(self.block_size) for r in self.waiting)
        if (waiting_blocks>=self.num_gpu_blocks*self.waiting_block_prop_threshold or
                occupied+request.blocks(self.block_size)>self.num_gpu_blocks or
                request.blocks(self.block_size,prompt_only=True)>free_gpu_blocks):return None
        for offset in range(self.pipeline_parallel_size):
            virtual=(self.cur_index+offset)%self.pipeline_parallel_size;existing=self._virtual_requests(virtual)
            if (len(existing)>=self.max_batch_size or
                sum(r.context_tokens for r in existing)+request.context_tokens>self.max_tokens_per_batch or
                sum(r.blocks(self.block_size) for r in existing)+request.blocks(self.block_size)>
                    self.num_gpu_blocks//self.pipeline_parallel_size or
                request.blocks(self.block_size,prompt_only=True)>free_blocks_by_virtual[virtual]):continue
            self.bridge.popleft();request.target_virtual=virtual;self.waiting.append(request)
            self.cur_index=(virtual+1)%self.pipeline_parallel_size
            return request
        return None

    def next_batch(self):
        while self.waiting:
            request=self.waiting[0]
            if request.cancelled:self.waiting.popleft();continue
            queue=self.batch_queues[request.target_virtual]
            if (len(queue)>=self.max_batch_size or
                sum(r.context_tokens for r in queue.values())+request.context_tokens>self.max_tokens_per_batch):break
            self.waiting.popleft();queue[request.request_id]=request;self.active[request.request_id]=request
        return tuple(self.active.values())

    def finish(self,request_id):
        super().finish(request_id)
        for queue in self.batch_queues:queue.pop(request_id,None)
