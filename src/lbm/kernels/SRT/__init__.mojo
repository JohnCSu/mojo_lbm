"""Defines the deprecated single-buffer SRT LBM kernel and its benchmarks.

The `LBM_kernel` here is kept for reference and benchmarking; new code should
use `double_buffer_kernel` from `src/lbm/kernels/double_buffer/`.
"""
from .LBM_gpu_kernel import LBM_kernel
