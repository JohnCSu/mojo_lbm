"""Defines general-purpose utility structs used across the solver.

The types in this module support the LBM implementation by providing
stack-allocated value-semantic vectors and host/device buffer containers that
keep their CPU and GPU views in sync.
"""
from .contextTileTensor import ContextTileTensor
from .vector import Vector
