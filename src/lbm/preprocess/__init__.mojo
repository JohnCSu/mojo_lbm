"""Provides setup helpers that prepare the distribution and boundary fields.

The functions in this module initialize the distribution function `f` from
analytic velocity fields and apply boundary conditions to the exterior walls
of the domain before the time-stepping kernel is launched.
"""
from .initial_condition import initialize_f_from_func, initialize_fluid_at_rest
