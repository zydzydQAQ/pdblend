"""DistServe-only per-request shape evidence; engine scheduling is unchanged."""
from pdblend_runtime.native_v1 import NativeWorker


class StageProfileWorker(NativeWorker):
    def _native_shape(self, output):
        shape = super()._native_shape(output)
        ids = shape['request_ids']
        shape.update(prompt_lengths=[self._native_requests[rid]['prompt'] for rid in ids],
                     context_lengths=[self._native_requests[rid]['computed'] for rid in ids],
                     scheduled_lengths=[output.num_scheduled_tokens[rid] for rid in ids])
        return shape
