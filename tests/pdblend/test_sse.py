from pdblend.proxy.sse import StreamScan

EV = b'data: {"id":"x","choices":[{"index":0,"text":"%s","finish_reason":null}],"usage":null}\n\n'
USAGE = b'data: {"id":"x","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}\n\n'
DONE = b"data: [DONE]\n\n"


def stream(texts):
    return b"".join(EV % t for t in texts) + USAGE + DONE


def test_counts_tokens_and_usage_whole_stream():
    s = StreamScan()
    fwd, n = s.feed(stream([b"a", b"b", b"", b"\\\"q"]))
    assert n == 3 and s.tokens == 3 and s.done
    assert fwd == stream([b"a", b"b", b"", b"\\\"q"])[:-len(DONE)]
    assert s.usage_tokens() == 7 and s.completion_tokens() == 7


def test_every_chunk_boundary_gives_same_count():
    spaced = b'data: {"choices": [{"text": "%s", "index": 0}]}\n\n'
    for ev in (EV, spaced):
        data = b"".join(ev % t for t in (b"hello", b"", b" world", b"!")) + USAGE + DONE
        for cut in range(1, len(data)):
            for cut2 in range(cut, len(data), 7):
                s = StreamScan()
                total = sum(s.feed(part)[1] for part in (data[:cut], data[cut:cut2], data[cut2:]))
                assert total == 3, (cut, cut2)
                assert s.done and s.usage_tokens() == 7


def test_without_usage_falls_back_to_counted_tokens():
    s = StreamScan()
    s.feed(b"".join(EV % t for t in (b"x", b"y")) + DONE)
    assert s.usage_tokens() is None and s.completion_tokens() == 2
