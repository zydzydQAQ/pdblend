# KV golden diagnosis without changing acceptance

`diagnose_kv_golden.py` only reads artifacts and writes a report plus exact
historical input token lists. It does not launch an engine, use a GPU, retokenize
legacy output text, or qualify a topology.

```bash
PYTHONPATH=src /home/pdblend/.venv/bin/python scripts/diagnose_kv_golden.py \
  results/2026-09-22/three-model/queue-attempts \
  --out results/2026-09-23/kv-golden-diagnostics
```

The eight historical PP1 artifacts have the same recorded source hash and image
digest. The 7168-token prompt produces two alternative texts that exchange the
mixed/P-D positions between TP2 and TP4 for both 7B and 32B. A 14B TP4 mismatch
also occurs at 512 tokens. These artifacts retain text, not generated token IDs,
logits, or KV digests, so they cannot locate the first differing **token** or
establish whether its cause is transfer damage or numerical variation. The
exported reproducer contains the two distinct failing input prompts, 512 and
7168 tokens, and every corresponding model/TP failure. The 2048-token control
uses the same generator with seed 204800.

The current connector saves all L prompt slots and loads all L slots. Its
`get_num_new_matched_tokens` advertises L−1, requiring the decoder to compute the
last prompt token again. This is not evidence of a missing transferred token:
it gives the first decode computation query length 1, whereas ordinary prefill
has query length L. The prefill engine's already sampled first output token is
discarded by the old client. For these single-request cases L≤7168 and the
configured token budget is 8192; actual scheduled-token metadata should verify
whether chunking occurred, instead of assuming all long inputs were chunked.

The vLLM 0.10.1.1 [sampler implementation](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.1.1/vllm/v1/sample/sampler.py)
returns argmax directly for greedy sampling. A seed does not repair a change in
logit ordering caused by floating-point arithmetic. The version's
[attention backend](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.1.1/vllm/v1/attention/backends/flash_attn.py)
uses query-length-dependent scheduling. This makes numerical-path variation a
specific hypothesis, not a proven diagnosis of the historical failures.

Run a bounded correctness probe while each model/TP pair is already resident:

1. Use the original failing token list, temperature 0, explicit seed 701, 16
   output tokens, the same TP and generation on P and D, and no competing
   requests. Repeat each identical prompt three times. Do not increment the
   prompt-generator seed between these repetitions. Keep 512/2048/7168 controls
   separate from these identical-input repetitions.
2. Save all three D ordinary references, P's one-token prefill output, and each
   D P-D output as token IDs. Store actual prompt IDs/checksum, request/transaction
   IDs, generation, rank/layer load receipts and engine arguments. The CLI reports
   the first differing zero-based token index/IDs, P first token versus D reference
   first token, and stability within each path. A Boolean `tokens_match` cannot
   replace the underlying IDs.
3. Enable the optional `pdblend_runtime.kv_digest` helper on this correctness
   probe only. Compare source-gathered KV, the received payload, and immediately
   injected destination slots for every layer and rank. Use the same target D
   request ID, transaction and generation for all three receipts. Source and
   destination physical block IDs may differ; logical payload bytes must match.
4. If bytes differ, investigate gather, transfer, spill, injection and lifetime
   before discussing numerical behavior. If bytes match but token IDs differ,
   record first-step/full-prefix logits at the first divergence with the same
   preceding token IDs (teacher forcing), including the two competing token
   logits and their margin. Comparisons after divergent prefixes do not measure
   the original discrepancy. A small logit margin is diagnostic, never a pass.
5. A separately identified carry-first-token variant can use P's actual first
   output ID in D's prompt, load all L original prompt KV slots, request 15
   remaining outputs, and compare the concatenated 16 IDs strictly. This avoids
   recomputing the prompt's last token and preserves ordinary full-prefill/decode
   semantics. It must record P's token arrival as the first output for TTFT,
   explicitly distinguish source L versus target prompt L+1, and retain native
   cancellation/release checks. Passing this variant does not qualify the old
   request protocol or a different graph/eager configuration.

For P2P's non-MLA path the transmitted payload is
`[2, token_count, flattened_head_dimension]`. PUTS concatenates these payloads and
restores each recorded shape on receipt. A pinned-memory spill stores an opaque
`(address, dtype, shape)` descriptor; this is not tensor evidence. Digest the
actual tensor after `recv_tensor` finishes its H2D copy. The helper views BF16 as
bytes without a float conversion, hashes contiguous logical order, and rejects
empty, invalid or stale receipts. Its blocking CPU copy is excluded from timing
and power measurements. The caller must honor producer CUDA-stream dependencies.

Matching one tensor's bytes only proves that inspected payload's transport.
Whole-request layer/rank completeness, correct token/slot identity, no leaked KV,
cancel/recovery, generations and exact output golden all remain required.
