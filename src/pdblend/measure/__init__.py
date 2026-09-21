# -*- coding: utf-8 -*-
"""测量公共库:锁频、功率、操作点表、prompts。替代已删的 motivations/common。"""
from pdblend.measure.backends import (
    BackendError, FakeBackend, GpuBackend, PynvmlBackend, get_backend,
)
from pdblend.measure.datasets import Dataset, TestRequest
from pdblend.measure.engine import make_llm, run_generate
from pdblend.measure.gpu_clock import (
    check_gpu_present, current_sm_clock, nearest_freq, reset_all_gpus,
    set_all_gpus_clock,
)
from pdblend.measure.perfmodel import PerfModel, _interp1d
from pdblend.measure.power import (
    PowerSampler, trapezoid_energy, trapezoid_mean_power,
)
from pdblend.measure.prompts import make_prompts

__all__ = [
    "BackendError", "Dataset", "FakeBackend", "GpuBackend", "PerfModel",
    "PowerSampler", "PynvmlBackend", "TestRequest", "_interp1d",
    "check_gpu_present", "current_sm_clock", "get_backend",
    "make_llm", "make_prompts", "nearest_freq", "reset_all_gpus",
    "run_generate",
    "set_all_gpus_clock", "trapezoid_energy", "trapezoid_mean_power",
]
