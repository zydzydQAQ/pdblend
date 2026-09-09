# -*- coding: utf-8 -*-
"""测量公共库:锁频、功率、操作点表、prompts。替代已删的 motivations/common。"""
from ecopadg.measure.backends import (
    BackendError, FakeBackend, GpuBackend, PynvmlBackend, get_backend,
)
from ecopadg.measure.csvio import append_csv, write_csv
from ecopadg.measure.datasets import Dataset, TestRequest
from ecopadg.measure.engine import make_llm, run_generate
from ecopadg.measure.gpu_clock import (
    check_gpu_present, current_sm_clock, nearest_freq, reset_all_gpus,
    set_all_gpus_clock,
)
from ecopadg.measure.perfmodel import PerfModel, _interp1d
from ecopadg.measure.power import (
    PowerSampler, trapezoid_energy, trapezoid_mean_power,
)
from ecopadg.measure.prompts import make_prompts

__all__ = [
    "BackendError", "Dataset", "FakeBackend", "GpuBackend", "PerfModel",
    "PowerSampler", "PynvmlBackend", "TestRequest", "_interp1d",
    "append_csv", "check_gpu_present", "current_sm_clock", "get_backend",
    "make_llm", "make_prompts", "nearest_freq", "reset_all_gpus",
    "run_generate",
    "set_all_gpus_clock", "trapezoid_energy", "trapezoid_mean_power",
    "write_csv",
]
