# -*- coding: utf-8 -*-
"""CSV 追加/覆盖。"""
from __future__ import annotations

import csv
import os
from typing import Iterable, Mapping, Sequence


def write_csv(path: str, fields: Sequence[str],
              rows: Iterable[Mapping]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def append_csv(path: str, fields: Sequence[str],
               rows: Iterable[Mapping]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)
