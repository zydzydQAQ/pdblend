"""Short-horizon load forecast from proxy request records."""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class Forecast:
    rate_rps: float          # arrivals per second (EWMA, short window)
    trend_rps: float         # short EWMA minus long EWMA; > 0 means rising load
    input_mean: float
    input_p95: float
    output_mean: float
    inflight: int
    inputs: tuple = ()       # recent input lengths, for split ratios by threshold
    outputs: tuple = ()      # recent output lengths, for the mixed-pool TPOT tail

    @property
    def samples(self) -> int:
        return len(self.inputs)

    def split(self, threshold: int) -> tuple[float, float, float]:
        """(share of requests with input >= threshold, mean input of that part, mean input of the rest)."""
        if threshold <= 0 or not self.inputs:
            return 1.0, self.input_mean, self.input_mean
        hi = [n for n in self.inputs if n >= threshold]
        lo = [n for n in self.inputs if n < threshold]
        return (len(hi) / len(self.inputs), sum(hi) / len(hi) if hi else self.input_mean,
                sum(lo) / len(lo) if lo else self.input_mean)


def percentile(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    k = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return float(values[k])


class Forecaster:
    """Bins arrivals per second; EWMA rate over ~short_s and ~long_s; length stats over window_s."""

    def __init__(self, short_s: float = 30.0, long_s: float = 120.0, window_s: float = 120.0,
                 default_input: float = 512.0, default_output: float = 128.0):
        self.alpha_short = 1.0 / short_s
        self.alpha_long = 1.0 / long_s
        self.window_s = window_s
        self.short = self.long = 0.0
        self.bin_start = None
        self.bin_count = 0
        self.arrivals: deque = deque()      # (t, input_tokens)
        self.completions: deque = deque()   # (t, output_tokens)
        self.default_input, self.default_output = default_input, default_output
        self.inflight = 0

    def arrive(self, input_tokens: int, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._advance(now)
        self.bin_count += 1
        self.inflight += 1
        self.arrivals.append((now, input_tokens))

    def finish(self, output_tokens: int, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.inflight = max(0, self.inflight - 1)
        if output_tokens > 0:
            self.completions.append((now, output_tokens))

    def _advance(self, now: float) -> None:
        if self.bin_start is None:
            self.bin_start = math.floor(now)
            return
        while self.bin_start + 1 <= now:
            self.short += self.alpha_short * (self.bin_count - self.short)
            self.long += self.alpha_long * (self.bin_count - self.long)
            self.bin_count = 0
            self.bin_start += 1
        cutoff = now - self.window_s
        for dq in (self.arrivals, self.completions):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def recent_inputs(self) -> list[int]:
        return [n for _, n in self.arrivals]

    def forecast(self, now: float | None = None) -> Forecast:
        now = time.time() if now is None else now
        self._advance(now)
        inputs = self.recent_inputs()
        outputs = [n for _, n in self.completions]
        # Blend the EWMA with the raw recent count so a cold start is not stuck at zero.
        recent = len(self.arrivals) / min(self.window_s, max(now - self.arrivals[0][0], 1.0)) if self.arrivals else 0.0
        rate = max(self.short, 0.5 * (self.short + recent)) if self.short else recent
        return Forecast(rate_rps=rate, trend_rps=self.short - self.long,
                        input_mean=sum(inputs) / len(inputs) if inputs else self.default_input,
                        input_p95=percentile(inputs, 0.95) if inputs else self.default_input,
                        output_mean=sum(outputs) / len(outputs) if outputs else self.default_output,
                        inflight=self.inflight, inputs=tuple(inputs), outputs=tuple(outputs))
