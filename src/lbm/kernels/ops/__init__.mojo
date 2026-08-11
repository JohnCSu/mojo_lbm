"""Defines LBM collision operators and boundary-condition kernels.

Provides BGK (SRT), two-relaxation-time (TRT), regularized (RLBM), and
entropy-stabilized (KBC) collision schemes, together with bounce-back
and equilibrium boundary conditions.
"""
from .boundary_condition import wall_bc,equilibrium_bc
from .collision import SRT,TRT,RLBM
from .KBC_ import KBC

