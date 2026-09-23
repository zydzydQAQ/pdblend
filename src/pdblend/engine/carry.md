# Public OpenAI first-token carry

The P/D path now retains the token actually sampled by P. A request with L prompt
IDs and an N-token budget sends P the original prompt with a one-token budget,
then D receives those L IDs plus the returned first ID and a budget of N−1.
The proxy emits P's first SSE event before starting D, and reports public usage
as L prompt tokens and N completion tokens. P's length-limited finish reason is
not forwarded as a completed logical request. Both legs use one public response
ID. A one-token request runs ordinary generation on P without a remote-KV ID or
handoff; it never starts a zero-token decode request.

`pd_complete(..., token_diagnostics=True)` returns `(prefill, combined)`.
`combined.token_ids` contains all actual generated IDs. Its TTFT starts at the P
submission and ends when P's response arrives. Raw D fields remain available as
`decode_submitted_s`, `decode_first_token_s`, and `decode_completion_tokens`.
The combined text is a diagnostic; exact IDs determine the output golden.

Ordinary `EngineClient.complete(..., token_diagnostics=True)` also captures exact
IDs. For the proxy, additionally pass `pdblend_token_diagnostics=True`; raw HTTP
clients set that field in the request JSON. The proxy exposes
`X-PDBlend-Token-Diagnostics` and `X-PDBlend-PD-Protocol` headers. Diagnostic mode
requests chosen-token logprobs on all engines and records the corresponding ID
strings. It is functional evidence and its timing/power cannot be substituted
for the default service path.

The default P leg requests `logprobs=0, return_tokens_as_token_ids=true`, which is
necessary to obtain its real first ID through pinned vLLM 0.10.1.1's public API.
The default D leg does **not** request logprobs. The necessary P overhead remains
part of real service cost; no cost is subtracted to manufacture a comparison.
No tokenizer is invoked to recover an ID from generated text.

Current carry support is intentionally bounded to flat token-ID prompts,
greedy sampling, one choice, `ignore_eos=true`, and no stop/history-dependent
sampling or caller-supplied logprob contract. The registered Qwen2.5 7B/14B/32B
tokenizer configurations mark IDs 151643 through 151656 as special; these are
the explicit default `PDTransfer.special_token_ids`, overridable for another
verified model. With `ignore_eos=true`, EOS is a valid carried token. Empty text
from a skipped special token is supported and its first SSE event includes an
explicit generated-token marker so TTFT still counts its arrival. Other empty
or replacement-character first-token text is rejected in the default path
because independent detokenizer boundaries are not proven equivalent;
diagnostics may retain these IDs while treating the text as non-authoritative.

Any malformed first-token evidence, missing final decode usage, extra choice,
wrong prompt count, output-budget mismatch, or diagnostic ID-count mismatch
fails closed. A post-prefill error may already have produced remote KV; it is
not an acknowledged native cleanup. Resident pools retain their existing error
quarantine behavior. Separate GPU cancel/cleanup proof is still required.

Old transfer measurements that inferred KV duration from logical TTFT are not
valid after this change: logical TTFT now correctly ends at P's token. Fresh
transfer/KV-load receipts and the distinct P-to-D continuation gap must be
recorded separately. Existing frozen measurements and running jobs are unchanged.

Protocol references: [pinned vLLM request fields](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/entrypoints/openai/protocol.py)
and [pinned completion/logprob serialization](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/entrypoints/openai/serving_completion.py).
