"""Read-only failure attribution for an immutable comparison window.

Categories partition canonical failed requests.  The cancellation, timeout and
rejection totals are independent flags and may overlap; a cohort timeout is
both a cancellation and a timeout.  A generic TimeoutError does not establish
that the predictor (or the serving engine) caused it.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
from typing import Iterator


CATEGORIES = (
    "native_reject", "native_timeout", "predictor_timeout", "timeout_unknown",
    "cohort_timeout_cancelled", "cancelled", "invalid_output", "unknown",
    "unresolved",
)
_REQUEST_ID = re.compile(r"(?:r(\d+)|(?:dynamo|dynamollm|distserve|ecoserve)-\d+-(\d+))\Z")


def _rows(path: Path) -> Iterator[tuple[int, dict]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if ".jsonl" in path.name:
            for line, text in enumerate(stream, 1):
                if text.strip():
                    value = json.loads(text)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line}: expected an object")
                    yield line, value
        else:
            value = json.load(stream)
            if isinstance(value, dict):
                value = value.get("outcomes", [])
            if not isinstance(value, list):
                raise ValueError(f"{path}: expected an outcomes list")
            for row in value:
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: expected outcome objects")
                yield 0, row


def _error(row: dict) -> str | None:
    value = row.get("error")
    if not value and isinstance(row.get("result"), dict):
        value = row["result"].get("error")
    return str(value) if value else None


def _is_timeout(text: str) -> bool:
    return any(value in text for value in ("timeout", "timed out", "deadline exceeded", "slo expired"))


def _explicit_predictor_timeout(row: dict) -> bool:
    text = " ".join([_error(row) or "", *(str(row.get(key, "")) for key in
                    ("kind", "event", "stage", "component", "status"))]).lower()
    return ("predictor" in text or "prediction" in text) and _is_timeout(text)


def analyze_failures(window_dir: Path, canonical_requests: list[dict]) -> dict:
    """Attribute canonical failures using request-aligned native evidence.

    ``window_dir`` may be the window directory or its ``run`` directory.
    Original files are never modified.  Malformed evidence raises an error
    instead of silently changing the denominator or fabricating attribution.
    """
    window_dir = Path(window_dir).resolve()
    run = window_dir / "run" if (window_dir / "run").is_dir() else window_dir
    canonical = {}
    identifiers = {}
    for row in canonical_requests:
        idx = row.get("idx")
        if type(idx) is not int or idx < 0 or idx in canonical:
            raise ValueError("canonical request indices must be unique nonnegative integers")
        canonical[idx] = row
        if row.get("request_id") is not None:
            rid = str(row["request_id"])
            if rid in identifiers:
                raise ValueError("duplicate canonical request ID")
            identifiers[rid] = idx
    failed = {idx: row for idx, row in canonical.items() if row.get("successful") is not True}
    sources: list[str] = []
    raw: dict[int, tuple[dict, str]] = {}
    cancellations: dict[int, tuple[bool, str]] = {}
    predictor_timeouts: dict[int, str] = {}

    def request_index(row: dict) -> int | None:
        explicit = row.get("idx")
        index = explicit if type(explicit) is int else None
        rid = row.get("request_id")
        if rid is not None:
            rid = str(rid)
            match = _REQUEST_ID.fullmatch(rid)
            identified = identifiers.get(rid)
            if identified is None and match:
                identified = int(next(group for group in match.groups() if group is not None))
            if index is not None and identified is not None and index != identified:
                raise ValueError(f"conflicting raw request index and ID: {rid}")
            index = identified if index is None else index
        return index if index in failed else None

    def remember(row: dict, source: str) -> None:
        idx = request_index(row)
        if idx is None:
            return
        previous = raw.get(idx)
        # Prefer the direct outcomes file; supplement missing errors from
        # completion/native results or explicit request events.
        if previous is None or (not _error(previous[0]) and _error(row)):
            raw[idx] = (row, source)
        if _explicit_predictor_timeout(row):
            predictor_timeouts[idx] = source

    if failed:
        for name in ("outcomes.jsonl", "outcomes.jsonl.gz", "outcomes.json",
                     "native-result.json", "completion.json"):
            path = run / name
            if path.is_file():
                sources.append(str(path))
                for line, row in _rows(path):
                    remember(row, f"{path}:{line}" if line else str(path))
        # Stream rather than materialize journals containing every output token.
        for name in ("events.jsonl.gz", "events.jsonl"):
            path = run / name
            if not path.is_file():
                continue
            sources.append(str(path))
            for line, row in _rows(path):
                event = row.get("kind", row.get("event", ""))
                source = f"{path}:{line} [{event}]"
                if event == "eco_comparison_cohort_cancel_begin":
                    timed_out = row.get("reason") == "cohort_timeout"
                    for request_id in row.get("pending_requests", []):
                        idx = request_index({"request_id": request_id})
                        if idx is not None:
                            previous = cancellations.get(idx)
                            if previous is None or timed_out:
                                cancellations[idx] = (timed_out, source)
                if (event in {"eco_request_outcome", "dynamo_outcome", "distserve_request_outcome"}
                        or _error(row) or _explicit_predictor_timeout(row)):
                    remember(row, source)

    counts = dict.fromkeys(CATEGORIES, 0)
    details = []
    cancelled_total = timeout_total = rejected_total = unresolved_total = 0
    for idx, row in failed.items():
        native, raw_source = raw.get(idx, ({}, "canonical_requests"))
        native_error = _error(native)
        canonical_error = _error(row)
        text = " ".join(filter(None, (native_error, canonical_error))).lower()
        cancellation = cancellations.get(idx)
        timed_out = _is_timeout(text) or row.get("timeout") is True
        rejected = any(word in text for word in ("reject", "bounded queue is full", "admission denied"))
        cancelled = "cancel" in text or cancellation is not None
        unresolved = canonical_error == "missing_outcome" or row.get("unresolved") is True
        source = raw_source if native_error else "canonical_requests"
        if cancellation is not None:
            cohort_timeout, source = cancellation
            category = "cohort_timeout_cancelled" if cohort_timeout else "cancelled"
            timed_out |= cohort_timeout
        elif idx in predictor_timeouts or _explicit_predictor_timeout(row):
            category = "predictor_timeout"
            source = predictor_timeouts.get(idx, "canonical_requests")
            timed_out = True
        elif cancelled:
            category = "cancelled"
        elif rejected:
            category = "native_reject"
        elif timed_out:
            category = ("native_timeout" if any(marker in text for marker in
                        ("admission slo expired", "request deadline exceeded", "engine request timeout"))
                        else "timeout_unknown")
        elif unresolved:
            category = "unresolved"
        elif any(marker in text for marker in ("output", "token", "timing", "terminal", "incomplete")):
            category = "invalid_output"
        else:
            category = "unknown"
        counts[category] += 1
        cancelled_total += int(cancelled)
        timeout_total += int(timed_out)
        rejected_total += int(rejected)
        unresolved_total += int(unresolved)
        details.append(dict(idx=idx, raw_error=native_error, canonical_error=canonical_error,
                            category=category, cause_source=source))
    return dict(counts=counts, cancelled_requests=cancelled_total, timeout_requests=timeout_total,
                rejected_requests=rejected_total, unresolved_requests=unresolved_total,
                failure_details=details, source_paths=sources,
                summary_semantics={
                    "counts": "mutually exclusive and exhaustive over canonical failed requests",
                    "totals": "cancellation/timeout/rejection/unresolved flags may overlap; cohort timeout counts as cancellation and timeout; cancel-begin alone does not resolve a missing outcome",
                    "timeout_unknown": "timeout observed; predictor/native stage is not established",
                })
