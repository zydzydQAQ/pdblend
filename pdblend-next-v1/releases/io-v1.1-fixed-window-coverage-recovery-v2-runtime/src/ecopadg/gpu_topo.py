# -*- coding: utf-8 -*-
"""GPU NUMA 拓扑与 P/D 段放置。

默认路径必须同岛配对 1P1D，禁止 PoolManager 那种先铺完 P 再铺 D。
探拓扑失败时退回段交错(与旧 run_cell 一致)，不退回 P-then-D。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ecopadg.types import Partition


def _world(tp: int, pp: int = 1) -> int:
    return max(int(tp), 1) * max(int(pp), 1)

REASON_NUMA = "numa-affinity"
REASON_INTERLEAVE = "interleave-fallback"


@dataclass
class MixedPlacement:
    gpus: List[int]
    numa: Optional[int]
    tp_cross_numa: bool


@dataclass
class SegmentPlacement:
    pgpus: List[int]
    dgpus: List[int]
    numa_p: Optional[int]
    numa_d: Optional[int]
    kv_cross_numa: bool


@dataclass
class Placement:
    mixed: List[MixedPlacement] = field(default_factory=list)
    segments: List[SegmentPlacement] = field(default_factory=list)
    extra_prefill: List[List[int]] = field(default_factory=list)
    extra_decode: List[List[int]] = field(default_factory=list)
    reason: str = REASON_NUMA
    numa_trusted: bool = True
    gpus: List[int] = field(default_factory=list)

    def as_assign(self) -> Dict[str, List[List[int]]]:
        """与历史 assign_gpus 相同的 {mixed,prefill,decode} 形状。"""
        pre = [list(s.pgpus) for s in self.segments] + [
            list(g) for g in self.extra_prefill]
        dec = [list(s.dgpus) for s in self.segments] + [
            list(g) for g in self.extra_decode]
        return {
            "mixed": [list(m.gpus) for m in self.mixed],
            "prefill": pre,
            "decode": dec,
        }

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["n_mixed"] = len(self.mixed)
        payload["n_segments"] = len(self.segments)
        return payload


def parse_gpu_list(spec: str) -> List[int]:
    text = str(spec).strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",") if x.strip() != ""]


def gpu_numa(idx: int) -> int:
    """PCI domain → NUMA；失败再 nvidia-smi topo；再退回 idx<4。"""
    node, trusted = probe_gpu_numa(idx)
    return node


def probe_gpu_numa(idx: int) -> Tuple[int, bool]:
    """返回 (numa, probed)。probed=False 表示只用了下标回退。"""
    try:
        bus = subprocess.check_output(
            ["nvidia-smi", "--id=%d" % int(idx),
             "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            text=True, timeout=8).strip().splitlines()[0].strip()
        if bus.startswith("00000001"):
            return 1, True
        if bus.startswith("00000000"):
            return 0, True
    except (OSError, subprocess.CalledProcessError, IndexError,
            subprocess.TimeoutExpired):
        pass
    try:
        topo = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, timeout=8)
        prefix = "GPU%d" % int(idx)
        for line in topo.splitlines():
            if not line.startswith(prefix):
                continue
            cols = line.split()
            for col in reversed(cols):
                if col.isdigit():
                    return int(col), True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return (0 if int(idx) < 4 else 1), False


def discover_numa(gpus: Sequence[int],
                  numa_of: Optional[Mapping[int, int]] = None
                  ) -> Tuple[Dict[int, int], bool]:
    """注入 numa_of 视为可信。否则逐卡探测；任一卡未探到则整体不可信。"""
    if numa_of is not None:
        return {int(g): int(numa_of[int(g)]) for g in gpus}, True
    out: Dict[int, int] = {}
    trusted = True
    for gpu in gpus:
        node, probed = probe_gpu_numa(int(gpu))
        out[int(gpu)] = int(node)
        trusted = trusted and probed
    return out, trusted


def _group_islands(gpus: Sequence[int],
                   numa: Mapping[int, int]) -> Dict[int, List[int]]:
    islands: Dict[int, List[int]] = defaultdict(list)
    for gpu in gpus:
        islands[int(numa[int(gpu)])].append(int(gpu))
    for node in islands:
        islands[node].sort()
    return dict(islands)


def _majority_numa(gpus: Sequence[int],
                   numa: Mapping[int, int]) -> Optional[int]:
    if not gpus:
        return None
    counts: Dict[int, int] = defaultdict(int)
    for gpu in gpus:
        counts[int(numa[int(gpu)])] += 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _take(pool: List[int], n: int) -> List[int]:
    if n <= 0:
        return []
    if len(pool) < n:
        raise ValueError("GPU 预算不足: need %d have %d" % (n, len(pool)))
    taken = pool[:n]
    del pool[:n]
    return taken


def _take_from_islands(islands: Dict[int, List[int]], n: int) -> List[int]:
    """优先从能一次拿满的岛取；否则按岛号拼，可能拆 TP。"""
    if n <= 0:
        return []
    for node in sorted(islands, key=lambda k: (-len(islands[k]), k)):
        if len(islands[node]) >= n:
            return _take(islands[node], n)
    taken: List[int] = []
    for node in sorted(islands, key=lambda k: (-len(islands[k]), k)):
        need = n - len(taken)
        if need <= 0:
            break
        grab = min(need, len(islands[node]))
        if grab:
            taken.extend(_take(islands[node], grab))
    if len(taken) < n:
        raise ValueError("GPU 预算不足: need %d have %d" % (n, len(taken)))
    return taken


def _numa_set(gpus: Sequence[int], numa: Mapping[int, int]) -> List[int]:
    return sorted({int(numa[int(g)]) for g in gpus})


def interleave_place(partition: Partition, gpus: Sequence[int],
                     numa: Optional[Mapping[int, int]] = None,
                     *, trusted: bool = False) -> Placement:
    """段交错: mixed 先切,再每段 P 然后 D。禁止先铺完所有 P。"""
    pool = [int(g) for g in gpus]
    numa_map = {int(g): int(numa[int(g)]) for g in gpus} if numa else {
        int(g): (0 if int(g) < 4 else 1) for g in gpus}
    mixed: List[MixedPlacement] = []
    for _ in range(int(partition.n_mixed)):
        grp = _take(pool, _world(partition.tp_mixed, partition.pp_mixed))
        nodes = _numa_set(grp, numa_map)
        mixed.append(MixedPlacement(
            gpus=grp, numa=nodes[0] if len(nodes) == 1 else None,
            tp_cross_numa=len(nodes) > 1))
    n_seg = min(int(partition.n_prefill), int(partition.n_decode))
    segments: List[SegmentPlacement] = []
    for _ in range(n_seg):
        pg = _take(pool, _world(partition.tp_prefill, partition.pp_prefill))
        dg = _take(pool, _world(partition.tp_decode, partition.pp_decode))
        pn = _majority_numa(pg, numa_map)
        dn = _majority_numa(dg, numa_map)
        segments.append(SegmentPlacement(
            pgpus=pg, dgpus=dg, numa_p=pn, numa_d=dn,
            kv_cross_numa=pn != dn))
    extra_p = [_take(pool, _world(partition.tp_prefill, partition.pp_prefill))
               for _ in range(int(partition.n_prefill) - n_seg)]
    extra_d = [_take(pool, _world(partition.tp_decode, partition.pp_decode))
               for _ in range(int(partition.n_decode) - n_seg)]
    return Placement(
        mixed=mixed, segments=segments,
        extra_prefill=extra_p, extra_decode=extra_d,
        reason=REASON_INTERLEAVE, numa_trusted=bool(trusted),
        gpus=[int(g) for g in gpus])


def _place_trusted(partition: Partition, gpus: Sequence[int],
                   numa: Mapping[int, int]) -> Placement:
    islands = _group_islands(gpus, numa)
    n_seg = min(int(partition.n_prefill), int(partition.n_decode))
    tp_p = _world(partition.tp_prefill, partition.pp_prefill)
    tp_d = _world(partition.tp_decode, partition.pp_decode)
    need = tp_p + tp_d
    segments: List[SegmentPlacement] = []
    for _ in range(n_seg):
        same = None
        for node in sorted(islands, key=lambda k: (-len(islands[k]), k)):
            if len(islands[node]) >= need:
                same = node
                break
        if same is not None:
            pg = _take(islands[same], tp_p)
            dg = _take(islands[same], tp_d)
            segments.append(SegmentPlacement(
                pgpus=pg, dgpus=dg, numa_p=same, numa_d=same,
                kv_cross_numa=False))
            continue
        p_node = next((n for n in sorted(
            islands, key=lambda k: (-len(islands[k]), k))
                       if len(islands[n]) >= tp_p), None)
        d_node = next((n for n in sorted(
            islands, key=lambda k: (-len(islands[k]), k))
                       if n != p_node and len(islands[n]) >= tp_d), None)
        if p_node is None or d_node is None:
            pg = _take_from_islands(islands, tp_p)
            dg = _take_from_islands(islands, tp_d)
        else:
            pg = _take(islands[p_node], tp_p)
            dg = _take(islands[d_node], tp_d)
        pn = _majority_numa(pg, numa)
        dn = _majority_numa(dg, numa)
        segments.append(SegmentPlacement(
            pgpus=pg, dgpus=dg, numa_p=pn, numa_d=dn,
            kv_cross_numa=pn != dn))
    extra_p = [_take_from_islands(islands, tp_p)
               for _ in range(int(partition.n_prefill) - n_seg)]
    extra_d = [_take_from_islands(islands, tp_d)
               for _ in range(int(partition.n_decode) - n_seg)]
    mixed: List[MixedPlacement] = []
    for _ in range(int(partition.n_mixed)):
        grp = _take_from_islands(
            islands, _world(partition.tp_mixed, partition.pp_mixed))
        nodes = _numa_set(grp, numa)
        mixed.append(MixedPlacement(
            gpus=grp, numa=nodes[0] if len(nodes) == 1 else None,
            tp_cross_numa=len(nodes) > 1))
    return Placement(
        mixed=mixed, segments=segments,
        extra_prefill=extra_p, extra_decode=extra_d,
        reason=REASON_NUMA, numa_trusted=True,
        gpus=[int(g) for g in gpus])


def place_partition(partition: Partition, gpus: Sequence[int],
                    numa_of: Optional[Mapping[int, int]] = None,
                    *, force_interleave: bool = False) -> Placement:
    """按 NUMA 放置;不可信或 force_interleave 时段交错。"""
    gpu_list = [int(g) for g in gpus]
    need = int(partition.total_gpus())
    if need > len(gpu_list):
        raise ValueError("分区超出 GPU 预算: %s > %d 卡"
                         % (partition, len(gpu_list)))
    numa, trusted = discover_numa(gpu_list, numa_of)
    if force_interleave or not trusted:
        return interleave_place(partition, gpu_list, numa, trusted=False)
    return _place_trusted(partition, gpu_list, numa)


def pick_pd_gpus(gpus: Sequence[int],
                 numa_of: Optional[Mapping[int, int]] = None,
                 *, cross: bool, tp: int = 2
                 ) -> Tuple[List[int], List[int], bool]:
    """给微基准选一对 P/D GPU。返回 (pgpus, dgpus, kv_cross_numa)。"""
    gpu_list = [int(g) for g in gpus]
    numa, trusted = discover_numa(gpu_list, numa_of)
    islands = _group_islands(gpu_list, numa)
    tp = max(int(tp), 1)
    if not cross:
        for node in sorted(islands, key=lambda k: (-len(islands[k]), k)):
            if len(islands[node]) >= 2 * tp:
                members = islands[node]
                return members[:tp], members[tp:2 * tp], False
        # 单岛不够 2*tp：同岛各取 1 卡
        for node in sorted(islands):
            if len(islands[node]) >= 2:
                return islands[node][:1], islands[node][1:2], False
    nodes = sorted(islands, key=lambda k: (-len(islands[k]), k))
    if len(nodes) >= 2 and len(islands[nodes[0]]) >= tp \
            and len(islands[nodes[1]]) >= tp:
        a, b = nodes[0], nodes[1]
        return islands[a][:tp], islands[b][:tp], True
    if len(gpu_list) >= 2 * tp:
        return gpu_list[:tp], gpu_list[tp:2 * tp], True
    raise ValueError("无法选出 P/D GPU 对")


def is_p_then_d(assign: Mapping[str, Sequence[Sequence[int]]],
                gpus: Sequence[int], n_segments: int, tp: int) -> bool:
    """识别旧 assign_gpus 的「先铺完 P 再铺 D」顺序。"""
    if n_segments < 2 or int(tp) <= 0:
        return False
    linear = [int(g) for g in gpus]
    pre = [list(map(int, grp)) for grp in assign.get("prefill", [])]
    dec = [list(map(int, grp)) for grp in assign.get("decode", [])]
    if len(pre) < 2 or len(dec) < 2:
        return False
    # 旧顺序: P0=gpus[0:tp] P1=gpus[tp:2tp] D0=gpus[2tp:3tp] D1=gpus[3tp:4tp]
    expect_p0 = linear[0:tp]
    expect_p1 = linear[tp:2 * tp]
    expect_d0 = linear[2 * tp:3 * tp]
    expect_d1 = linear[3 * tp:4 * tp]
    return (pre[0] == expect_p0 and pre[1] == expect_p1
            and dec[0] == expect_d0 and dec[1] == expect_d1)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="NUMA-aware P/D GPU placement")
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--n-mixed", type=int, default=0)
    ap.add_argument("--n-segments", type=int, default=2)
    ap.add_argument("--tp-mixed", type=int, default=2)
    ap.add_argument("--tp-p", type=int, default=2)
    ap.add_argument("--tp-d", type=int, default=2)
    ap.add_argument("--pp-mixed", type=int, default=1)
    ap.add_argument("--pp-p", type=int, default=1)
    ap.add_argument("--pp-d", type=int, default=1)
    ap.add_argument("--force-interleave", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    gpus = parse_gpu_list(args.gpus)
    n_seg = max(int(args.n_segments), 0)
    part = Partition(
        n_mixed=int(args.n_mixed),
        n_prefill=n_seg, n_decode=n_seg,
        tp_mixed=int(args.tp_mixed),
        tp_prefill=int(args.tp_p), tp_decode=int(args.tp_d),
        pp_mixed=int(args.pp_mixed),
        pp_prefill=int(args.pp_p), pp_decode=int(args.pp_d),
        gpus_total=len(gpus))
    placement = place_partition(
        part, gpus, force_interleave=bool(args.force_interleave))
    text = json.dumps(placement.to_json(), ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
