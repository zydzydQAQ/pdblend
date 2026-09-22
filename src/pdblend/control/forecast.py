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
    peak_rps: float = 0.0    # max completed 60 s bin rate over the trailing epoch (observed history only)
    completed_bins: int = 0  # completed 60 s bins in that window; 0 means no usable load template yet
    recent_rate_rps: float = 0.0  # raw arrival rate, for auditing the smoothed estimate

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
    """Bins arrivals per second; EWMA rate over ~short_s and ~long_s; length stats over window_s.

    Also folds arrivals into 60 s bins: the max completed bin rate over the trailing epoch_s is the
    observed-history load template (DynamoLLM ScaleInst), built strictly from past arrivals."""

    def __init__(self, short_s: float = 30.0, long_s: float = 120.0, window_s: float = 120.0,
                 default_input: float = 512.0, default_output: float = 128.0,
                 bin_s: float = 60.0, epoch_s: float = 1800.0,
                 initial: Forecast | None = None):
        self.alpha_short = 1.0 / short_s
        self.alpha_long = 1.0 / long_s
        self.window_s = window_s
        self.short = self.long = 0.0
        self.bin_start = None
        self.bin_count = 0
        self.bin_s, self.epoch_s = bin_s, epoch_s
        self.bin60_start = None
        self.bin60_count = 0
        self.peak_bins: deque = deque()     # (bin_end_t, arrivals_per_second) of completed 60 s bins
        self.arrivals: deque = deque()      # (t, input_tokens)
        self.completions: deque = deque()   # (t, output_tokens)
        self.default_input, self.default_output = default_input, default_output
        self.inflight = 0
        self.initial = initial
        self.traffic_start: float | None = None

    def arrive(self, input_tokens: int, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.traffic_start is None:
            self.traffic_start = now
            if self.initial is not None:
                # Start the prior when traffic arrives, not during engine/proxy setup.
                self.short = self.long = self.initial.rate_rps
                self.bin_start = math.floor(now)
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
            if self.bin60_start is None:
                self.bin60_start = self.bin_start
            self.bin60_count += self.bin_count
            if self.bin_start + 1 - self.bin60_start >= self.bin_s:
                self.peak_bins.append((self.bin60_start + self.bin_s, self.bin60_count / self.bin_s))
                self.bin60_start += self.bin_s
                self.bin60_count = 0
            self.bin_count = 0
            self.bin_start += 1
        cutoff = now - self.window_s
        for dq in (self.arrivals, self.completions):
            while dq and dq[0][0] < cutoff:
                dq.popleft()
        cutoff_bin = now - self.epoch_s
        while self.peak_bins and self.peak_bins[0][0] < cutoff_bin:
            self.peak_bins.popleft()

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
        output_mean = sum(outputs) / len(outputs) if outputs else self.default_output
        if self.initial is not None:
            age = max(0.0, now - self.traffic_start) if self.traffic_start is not None else 0.0
            prior_weight = max(0.0, 1.0 - age / self.window_s)
            # Early completions are length-biased (warmup requests are short too).
            # Fade out the existing warm-start prior over one observation window.
            if prior_weight > 0:
                observed_mean = output_mean if outputs else self.initial.output_mean
                output_mean = prior_weight * self.initial.output_mean + (1.0 - prior_weight) * observed_mean
                prior_outputs = self.initial.outputs or (self.initial.output_mean,)
                if outputs:
                    n_prior = round(64 * prior_weight)
                    ordered_prior, ordered_actual = sorted(prior_outputs), sorted(outputs)
                    def sample(values, n):
                        return [values[min(len(values) - 1, int((i + .5) * len(values) / n))] for i in range(n)]
                    outputs = sample(ordered_prior, n_prior) + sample(ordered_actual, 64 - n_prior)
                else:
                    outputs = list(prior_outputs)
            if self.traffic_start is None:
                rate = self.initial.rate_rps
        default_input = self.initial.input_mean if self.initial is not None else self.default_input
        default_p95 = self.initial.input_p95 if self.initial is not None else self.default_input
        return Forecast(rate_rps=rate, trend_rps=self.short - self.long,
                        input_mean=sum(inputs) / len(inputs) if inputs else default_input,
                        input_p95=percentile(inputs, 0.95) if inputs else default_p95,
                        output_mean=output_mean,
                        inflight=self.inflight, inputs=tuple(inputs), outputs=tuple(outputs),
                        peak_rps=max((r for _, r in self.peak_bins), default=0.0),
                        completed_bins=len(self.peak_bins), recent_rate_rps=recent)
