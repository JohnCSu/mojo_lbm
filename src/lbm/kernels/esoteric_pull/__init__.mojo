"""Defines the esoteric-pull LBM kernel.

This kernel uses the esoteric-pull streaming scheme to read and write
populations in-place, halving memory traffic compared with the double-buffer
variant. The implementation is incomplete.
"""
from .GPU_kernel import esoteric_pull_kernel