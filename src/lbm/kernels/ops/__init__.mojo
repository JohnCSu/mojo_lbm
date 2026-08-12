"""Defines LBM collision operators and boundary-condition kernels.

Provides BGK (SRT), two-relaxation-time (TRT), regularized (RLBM), and
entropy-stabilized (KBC) collision schemes, together with bounce-back
and equilibrium boundary conditions.
"""
from .boundary_condition import moving_wall_bc,equilibrium_bc
from .collisions import SRT,TRT,RLBM,KBC

