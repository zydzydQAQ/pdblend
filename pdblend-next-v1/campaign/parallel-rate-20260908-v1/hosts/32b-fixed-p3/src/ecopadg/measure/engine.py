# -*- coding: utf-8 -*-
"""vLLM LLM 构造:关 prefix cache,与动机实验协议一致。"""
from __future__ import annotations


def make_llm(model: str, tensor_parallel_size: int = 1,
             max_model_len: int = 8192,
             gpu_memory_utilization: float = 0.85,
             enable_prefix_caching: bool = False, **kwargs):
    from vllm import LLM

    return LLM(
        model=model,
        tensor_parallel_size=int(tensor_parallel_size),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enable_prefix_caching=bool(enable_prefix_caching),
        trust_remote_code=True,
        **kwargs,
    )


def run_generate(llm, prompts, max_tokens: int, use_tqdm: bool = False) -> float:
    """同步 generate,返回墙钟秒。"""
    import time
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=int(max_tokens), ignore_eos=True,
                            temperature=0.0)
    t0 = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=use_tqdm)
    return time.perf_counter() - t0
