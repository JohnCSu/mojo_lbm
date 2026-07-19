"""Defines the core LBM solver types, lattice models, and grid abstractions.

This package groups the domain model (grids, lattice models, unit systems),
preprocessing (boundary and initial conditions), geometry primitives, and the
GPU kernels that perform streaming, collision, and post-processing. A single
SRT kernel serves all dimensions and lattice models through compile-time
parameterization.
"""
from .constants import *
from .grid import LBM_Grid, GridLike
from .preprocess.boundary_condition import (
    set_exterior_walls,
    set_exterior_walls_with_func,
)
from .lattice_models import get_D2Q9, get_D3Q19, get_D3Q27, LatticeModel
from .kernels.output import calculate_rho_and_velocity,calculate_esoteric_rho_and_velocity
from .config import LBM_Config, ConfigLike
from .units import UnitSystem, Unit

# from .grid import GridLike
