# -*- coding: utf-8 -*-
"""定长 dummy prompt,关掉语义,只对齐 token 数。"""
from __future__ import annotations

from typing import Callable, List


def make_prompts(encode: Callable[[str], int], prompt_len: int,
                 batch: int, word: str = "hello") -> List[str]:
    """encode(prompt) ≈ prompt_len;同批重复同一条。"""
    target = max(int(prompt_len), 1)
    pieces = []
    n = 0
    while n < target:
        pieces.append(word)
        n = int(encode(" ".join(pieces)))
        if n >= target:
            break
        if len(pieces) > target * 4:
            break
    while n > target and pieces:
        pieces.pop()
        n = int(encode(" ".join(pieces))) if pieces else 0
    text = " ".join(pieces) if pieces else word
    return [text] * max(int(batch), 1)
