"""PDblend profiling-only shape receipts; scheduling and CUDA timing are inherited."""
from pdblend_runtime.native_v1 import NativeWorker


class PDNativeTimingWorker(NativeWorker):
    def _native_shape(self, output):
        shape = super()._native_shape(output)
        ids = shape['request_ids']
        shape.update(prompt_lengths=[self._native_requests[r]['prompt'] for r in ids],
                     context_lengths=[self._native_requests[r]['computed'] for r in ids],
                     scheduled_lengths=[output.num_scheduled_tokens[r] for r in ids])
        return shape
