# -*- coding: utf-8 -*-
from __future__ import annotations

from ecopadg.ecoserve_proxy import (
    BLOCK_SIZE, IMPLEMENTATION_LABEL, MEASURED_L20_14B_GPU_BLOCKS,
    MacroScheduler, count_completion_tokens, parse_vllm_kv_metrics,
    prompt_len_from_body, write_prefill_csv,
)


def _sched(n=2, ttft=5000, tpot=150, prefill=None, blocks=1000):
    data = prefill or {16: 10.0, 4096: 800.0}
    return MacroScheduler(n, ttft, tpot, data, num_gpu_blocks=blocks)


def test_empty_instance_accepts_first_request():
    sched = _sched()
    idx = sched.schedule("r0", 32)
    assert idx == 0
    assert sched.prefill_instance == 0
    assert sched.switch_count == 0
    assert "r0" in sched.instance_states[0].waiting_queue
    assert sched.instance_states[0].free_blocks == 1000 - 3


def test_init_uses_measured_blocks_not_1e9():
    sched = MacroScheduler(2, 5000, 150, {16: 10.0, 4096: 800.0},
                           num_gpu_blocks=0)
    assert sched.num_gpu_blocks == MEASURED_L20_14B_GPU_BLOCKS
    assert sched.instance_states[0].free_blocks == MEASURED_L20_14B_GPU_BLOCKS
    assert sched.instance_states[0].free_blocks < 10 ** 6


def test_kv_full_switches_instance():
    sched = _sched()
    sched.instance_states[0].free_blocks = 1
    idx = sched.schedule("r0", 64)
    assert idx == 1
    assert sched.switch_count == 1
    assert sched.send_output[0] is True
    assert sched.send_output[1] is False
    assert sched.prefill_instance == 1


def test_need_time_over_ttft_switches():
    sched = _sched(ttft=50, tpot=20, prefill={16: 80.0, 4096: 80.0})
    sched.schedule("r0", 16)
    idx = sched.schedule("r1", 16)
    assert idx == 1
    assert sched.switch_count == 1


def test_switch_cycles_prefill_instance():
    sched = _sched(n=3)
    sched.instance_states[0].free_blocks = 0
    sched.schedule("r0", 32)
    assert sched.prefill_instance == 1
    sched.instance_states[1].free_blocks = 0
    sched.schedule("r1", 32)
    assert sched.prefill_instance == 2


def test_block_size_matches_source():
    assert BLOCK_SIZE == 16
    sched = _sched()
    sched.schedule("r0", 17)
    assert sched.instance_states[0].requests[0].prefill_blocks == 2


def test_predict_time_interpolates_without_0p2n():
    sched = _sched(prefill={16: 10.0, 4096: 800.0})
    assert sched._get_predict_time(16) == 10
    mid = sched._get_predict_time(2048)
    assert 10 < mid < 800
    assert sched._get_predict_time(8) == 5


def test_write_prefill_csv_is_measured_tp1(tmp_path):
    path = tmp_path / "prefill.csv"
    write_prefill_csv(str(path), opmodel=None, tp=1)
    text = path.read_text(encoding="utf-8")
    assert "Length,Prefill Time" in text
    assert "512," in text
    rows = [ln.split(",") for ln in text.strip().splitlines()[1:]]
    by = {int(a): float(b) for a, b in rows}
    # 0.2×n @ 512 would be 102.4; measured TP1 is ~290 ms.
    assert by[512] > 200.0
    assert abs(by[512] / 512.0 - by[1024] / 1024.0) < 1e-6


def test_parse_vllm_kv_metrics():
    text = (
        "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage\n"
        "vllm:gpu_cache_usage_perc{model_name=\"qwen\"} 0.25\n"
        "vllm:cache_config_info{block_size=\"16\",num_gpu_blocks=\"15060\"} 1.0\n"
    )
    usage, blocks = parse_vllm_kv_metrics(text)
    assert usage == 0.25
    assert blocks == 15060
    sched = _sched(blocks=15060)
    sched.apply_kv_snapshot(0, int(round(15060 * 0.75)), 15060)
    assert sched.instance_states[0].free_blocks == 11295
    assert sched.kv_from_metrics is True


def test_prompt_len_uses_body_field():
    assert prompt_len_from_body({"prompt": "abcd" * 20, "prompt_len": 77}) == 77


def test_mark_output_tokens_increments_and_refreshes_clock():
    sched = _sched()
    sched.schedule("r0", 32)
    sched.mark_first_token(0, "r0")
    before = sched.instance_states[0].requests[0].num_iterations
    sched.instance_states[0].schedule_time = 1.0
    sched.mark_output_tokens(0, "r0", 3)
    assert sched.instance_states[0].requests[0].num_iterations == before + 3
    assert sched.instance_states[0].schedule_time > 1.0


def test_count_completion_tokens():
    raw = (
        b"data: {\"choices\":[{\"text\":\"Hello\"}]}\n\n"
        b"data: {\"choices\":[{\"text\":\" world\"}]}\n\n"
        b"data: [DONE]\n\n"
    )
    assert count_completion_tokens(raw) == 2


def test_implementation_label():
    assert IMPLEMENTATION_LABEL == "ecoserve-macro-8x-tp1-vllm092"
