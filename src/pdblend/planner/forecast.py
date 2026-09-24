"""Short-horizon load forecast from proxy request records."""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class InFlightWork:
    """Observed work bound to its current owner, including requests older than the sample window."""
    request_id: str
    input_tokens: int
    remaining_output_tokens: int
    waiting_prefill_tokens: int = 0
    kv_tokens: int = 0
    branch: str = ""
    pool_id: str = ""

    def __post_init__(self):
        if any(value < 0 for value in (self.input_tokens, self.remaining_output_tokens,
                                      self.waiting_prefill_tokens, self.kv_tokens)):
            raise ValueError("in-flight work must be nonnegative")
        if self.branch not in {"", "M", "PD", "P_ONLY"}:
            raise ValueError("unknown in-flight branch")


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

    # Arrival and completion deques are independent: never zip them into pairs.
    length_pairs: tuple[tuple[int, int], ...] = ()
    backlog: tuple[InFlightWork, ...] = ()

    @property
    def pending_prefill_tokens(self) -> int:
        return sum(work.waiting_prefill_tokens for work in self.backlog)

    @property
    def remaining_decode_tokens(self) -> int:
        return sum(work.remaining_output_tokens for work in self.backlog)

    @property
    def occupied_kv_tokens(self) -> int:
        return sum(work.kv_tokens for work in self.backlog)

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

    def split_forecasts(self, threshold: int) -> tuple["Forecast", "Forecast"]:
        """Conditional PD/M forecasts with correlated output lengths and bound in-flight work.

        Shares are measured from arrivals, avoiding the completion bias of short
        requests. Conditional output lengths use only explicitly paired samples;
        absent a paired observation, retain the global estimate.
        """
        share, _, _ = self.split(threshold)
        def branch(high: bool) -> Forecast:
            selected = lambda n: (threshold <= 0 or n >= threshold) == high
            ins = tuple(n for n in self.inputs if selected(n))
            pairs = tuple(pair for pair in self.length_pairs if selected(pair[0]))
            outs = tuple(pair[1] for pair in pairs) or self.outputs
            backlog = tuple(work for work in self.backlog
                            if ((work.branch in {"PD", "P_ONLY"}) == high if work.branch
                                else selected(work.input_tokens)))
            conditional_inputs = ins or tuple(work.input_tokens for work in backlog)
            fraction = share if high else 1.0 - share
            return replace(self, rate_rps=self.rate_rps * fraction,
                           trend_rps=self.trend_rps * fraction,
                           input_mean=sum(conditional_inputs) / len(conditional_inputs) if conditional_inputs else self.input_mean,
                           input_p95=percentile(conditional_inputs, .95) if conditional_inputs else self.input_p95,
                           output_mean=sum(pair[1] for pair in pairs) / len(pairs) if pairs else self.output_mean,
                           inputs=ins, outputs=outs, length_pairs=pairs, backlog=backlog,
                           inflight=len(backlog) if self.backlog else self.inflight,
                           peak_rps=self.peak_rps * fraction,
                           recent_rate_rps=self.recent_rate_rps * fraction)
        return branch(True), branch(False)

    def for_pool(self, pool_id: str, share: float) -> "Forecast":
        """Scale future arrivals while retaining work already bound to a resident pool."""
        if not 0 <= share <= 1:
            raise ValueError("pool share must lie in [0, 1]")
        backlog = tuple(work for work in self.backlog if work.pool_id == pool_id)
        return replace(self, rate_rps=self.rate_rps * share, trend_rps=self.trend_rps * share,
                       peak_rps=self.peak_rps * share, recent_rate_rps=self.recent_rate_rps * share,
                       backlog=backlog, inflight=len(backlog))


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
        self.paired_completions: deque = deque()  # (time, input, output), explicit request correlation
        self.pending_inputs: dict[str, int] = {}
        self.backlog: tuple[InFlightWork, ...] = ()
        self.completions: deque = deque()   # (t, output_tokens)
        self.default_input, self.default_output = default_input, default_output
        self.inflight = 0
        self.initial = initial
        self.traffic_start: float | None = None

    def arrive_request(self, input_tokens: int, max_tokens: int, *, request_id: str) -> None:
        self.arrive(input_tokens, request_id=request_id)

    def finish_request(self, output_tokens: int, *, request_id: str, input_tokens: int) -> None:
        self.finish(output_tokens, request_id=request_id, input_tokens=input_tokens)

    def set_backlog(self, items) -> None:
        """Replace the active snapshot; this data is never expired with length samples."""
        items = tuple(items)
        if len({item.request_id for item in items}) != len(items):
            raise ValueError("duplicate request in backlog snapshot")
        self.backlog = items

    def arrive(self, input_tokens: int, now: float | None = None, *,
               request_id: str | None = None) -> None:
        now = time.time() if now is None else now
        if request_id is not None:
            if request_id in self.pending_inputs:
                raise ValueError("request_id is already in flight")
            self.pending_inputs[request_id] = input_tokens
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

    def finish(self, output_tokens: int, now: float | None = None, *,
               request_id: str | None = None, input_tokens: int | None = None) -> None:
        now = time.time() if now is None else now
        paired_input = self.pending_inputs.pop(request_id, None) if request_id is not None else None
        if input_tokens is not None:
            if paired_input is not None and input_tokens != paired_input:
                raise ValueError("completion input length differs from its arrival")
            paired_input = input_tokens
        self.inflight = max(0, self.inflight - 1)
        if output_tokens > 0:
            self.completions.append((now, output_tokens))
            if paired_input is not None:
                self.paired_completions.append((now, paired_input, output_tokens))

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
        for dq in (self.arrivals, self.completions, self.paired_completions):
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
        pairs = [(i, o) for _, i, o in self.paired_completions]
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
                # Keep the bootstrap input/output association while fading it
                # out. Early completions otherwise replace long-output priors
                # with the shortest completed requests in each input branch.
                if self.initial.length_pairs:
                    if pairs:
                        n_prior = round(64 * prior_weight)
                        def pair_sample(values, count):
                            ordered = sorted(values)
                            return [ordered[min(len(ordered) - 1, int((i + .5) * len(ordered) / count))]
                                    for i in range(count)]
                        pairs = (pair_sample(self.initial.length_pairs, n_prior)
                                 + pair_sample(pairs, 64 - n_prior))
                    else:
                        pairs = list(self.initial.length_pairs)
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
                        completed_bins=len(self.peak_bins), recent_rate_rps=recent,
                        length_pairs=tuple(pairs),
                        backlog=self.backlog)
