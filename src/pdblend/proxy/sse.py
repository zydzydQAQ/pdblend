"""Byte-level SSE scanning: count token events and find usage without decoding every event as JSON."""
from __future__ import annotations

import re
from typing import Optional

DONE = b"data: [DONE]"
_TEXT = re.compile(rb'"text":\s*"(?=[^"])')       # non-empty text field (lookahead needs the char present)
_USAGE = re.compile(rb'"completion_tokens":\s*(\d+)')
_EMPTY_CARRIED_TOKEN = re.compile(rb'"pdblend_generated_tokens":\s*1(?=\s*[,}])')
_CARRY = 64


class StreamScan:
    """Feed raw chunks of a completions stream; counts non-empty text events across chunk boundaries."""

    def __init__(self):
        self.tokens = 0
        self.done = False
        self._carry = b""
        self._hold = b""
        self._tail = b""
        self._seen = 0            # absolute stream offset up to which content chars were already counted

    def feed(self, chunk: bytes) -> tuple[bytes, int]:
        """Returns (bytes before [DONE], number of new content events in this chunk)."""
        chunk = self._hold + chunk
        i = chunk.find(DONE)
        if i >= 0:
            chunk, self.done, self._hold = chunk[:i], True, b""
        else:
            # keep back a suffix that could be the start of a split [DONE]
            self._hold = next((chunk[-n:] for n in range(min(len(DONE) - 1, len(chunk)), 0, -1)
                               if DONE.startswith(chunk[-n:])), b"")
            chunk = chunk[:len(chunk) - len(self._hold)]
        buf, base = self._carry + chunk, self._seen - len(self._carry)
        n = sum(1 for m in _TEXT.finditer(buf) if base + m.end() >= self._seen)
        n += sum(1 for m in _EMPTY_CARRIED_TOKEN.finditer(buf) if base + m.end() >= self._seen)
        self._seen = base + len(buf)
        self._carry = buf[-_CARRY:]
        self._tail = (self._tail + chunk)[-512:]
        self.tokens += n
        return chunk, n

    def usage_tokens(self) -> Optional[int]:
        m = _USAGE.findall(self._tail)
        return int(m[-1]) if m else None

    def completion_tokens(self) -> int:
        return self.usage_tokens() or self.tokens
