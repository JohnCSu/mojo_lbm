"""Provides the GPU kernels for LBM streaming, collision, and diagnostics.

The kernels module contains the time-stepping implementations (double-buffer
and esoteric-pull), shared utility functions for moments, equilibrium, finite
differences, and shared-memory tile loading, plus post-processing kernels for
density, velocity, Q-criterion, and drag.
"""
from .double_buffer import double_buffer_kernel
