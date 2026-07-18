"""Defines the double-buffer SRT LBM kernel.

The kernel reads populations from `f_in` and writes the post-collision
populations to `f_out`, allowing the caller to swap buffers between time
steps without an in-place dependency.
"""
from .GPU_kernel import double_buffer_kernel
