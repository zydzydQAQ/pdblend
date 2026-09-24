"""Byte-level SSE scanning: count token events and find usage without decoding every event as JSON."""
from __future__ import annotations

import re
from typing import Optional
from pdblend.engine.carry import CarryProtocolError, SSEEvents

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
        usage = self.usage_tokens()
        return self.tokens if usage is None else usage


class TerminalStream:
    """Validate native completion separately from approximate token counting.

    EOF and [DONE] alone do not establish successful generation. Full-budget
    usage plus DONE is sufficient; shorter output requires an explicit legal
    stop reason (EOS and caller stop sequences both use ``stop``).
    """

    def __init__(self, max_tokens: int, *, prompt_tokens=None, choices: int = 1):
        self.parser = SSEEvents()
        self.max_tokens, self.prompt_tokens, self.choices = max_tokens, prompt_tokens, choices
        self.usage = None
        self.reasons = {}

    def feed(self, chunk):
        for event in self.parser.feed(chunk):
            if event.get("error"):
                raise CarryProtocolError(f"upstream stream error: {event['error']}")
            for choice in event.get("choices", ()):
                index = choice.get("index", 0)
                if type(index) is not int or not 0 <= index < self.choices:
                    raise CarryProtocolError("unexpected upstream choice index")
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason not in ("length", "stop"):
                        raise CarryProtocolError(f"upstream unsuccessful finish reason: {reason}")
                    self.reasons[index] = reason
            if event.get("usage") is not None:
                self.usage = event["usage"]

    def completion_tokens(self):
        if not self.parser.done or not isinstance(self.usage, dict):
            raise CarryProtocolError("upstream stream lacks terminal DONE and usage")
        n = self.usage.get("completion_tokens")
        if type(n) is not int or not 0 <= n <= self.max_tokens * self.choices:
            raise CarryProtocolError("upstream completion usage exceeds request token budget")
        prompt = self.usage.get("prompt_tokens")
        if ((prompt is not None and (type(prompt) is not int or prompt < 0))
                or (self.prompt_tokens is not None and prompt != self.prompt_tokens)):
            raise CarryProtocolError("upstream prompt usage disagrees with request")
        if prompt is not None and self.usage.get("total_tokens", prompt + n) != prompt + n:
            raise CarryProtocolError("upstream total token usage is inconsistent")
        if n < self.max_tokens * self.choices and (len(self.reasons) != self.choices or
                                                  "stop" not in self.reasons.values()):
            raise CarryProtocolError("short upstream completion lacks an explicit EOS/stop")
        return n
