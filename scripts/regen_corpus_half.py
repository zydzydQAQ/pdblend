#!/usr/bin/env python3
"""Derive a reduced-input corpus from an existing prepared corpus by middle-truncating token ids.

Keeps every record and split structure; only prompts longer than the per-dataset cap are
shortened (head cap//2 + tail cap-cap//2, same rule as pdblend.serving.datasets.encode_workload).
output_tokens are reference lengths and stay untouched.
"""
import hashlib
import json
import sys
from pathlib import Path

CAPS = {"sharegpt": 1024, "longbench": 3072}
SPLITS = ("calibration", "tuning", "evaluation")


def truncate(ids, cap):
    if len(ids) <= cap:
        return ids, False
    head = cap // 2
    return ids[:head] + ids[-(cap - head):], True


def shape_sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def main(src_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {}
    for js in sorted(src_root.glob("*.json")):
        if js.stem.endswith("-selection") or js.stem == "manifest":
            continue
        data = json.loads(js.read_text())
        ds = data.get("dataset", js.stem)
        cap = CAPS.get(ds)
        n_trunc = 0
        for split in SPLITS:
            for rec in data.get(split, []):
                if cap is None:
                    continue
                ids, did = truncate(rec["prompt"], cap)
                if did:
                    rec["prompt"] = ids
                    rec["input_tokens"] = len(ids)
                    rec["truncated"] = True
                    rec["content_sha256"] = shape_sha(ids)
                    rec["request_shape_sha256"] = rec["content_sha256"]
                    n_trunc += 1
        data["derived_from"] = str(src_root)
        if cap:
            data["input_cap"] = cap
        (out_root / js.name).write_text(json.dumps(data))
        lens = [r["input_tokens"] for r in data["evaluation"]]
        lens.sort()
        stats[ds] = dict(n_eval=len(lens), truncated=n_trunc,
                         mean=round(sum(lens) / len(lens), 1),
                         p95=lens[int(0.95 * (len(lens) - 1))], max=lens[-1])
    manifest = dict(schema=1, derived_from=str(src_root), input_caps=CAPS,
                    note="middle-truncated token ids; records/splits/order preserved; output_tokens untouched",
                    datasets=stats)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
