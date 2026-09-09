# -*- coding: utf-8 -*-
"""DistServe 风格 .ds:Dataset + TestRequest,pickle 落盘。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List


@dataclass
class TestRequest:
    prompt: str = ""
    prompt_len: int = 0
    output_len: int = 0


TestRequest.__test__ = False


@dataclass
class Dataset:
    dataset_name: str = ""
    reqs: List[TestRequest] = field(default_factory=list)

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "Dataset":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        raise TypeError("不是 Dataset: %r" % type(obj))
