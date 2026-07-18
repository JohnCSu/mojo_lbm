"""Provides shared helpers used by the LBM GPU kernels.

The functions in this module implement the reusable pieces of an LBM step:
index arithmetic, distribution-function load and store with optional Float16C
conversion, moment extraction (density, velocity, strain rate), equilibrium
and non-equilibrium populations, Smagorinsky LES, finite-difference velocity
gradients, and shared-memory tile loading with a halo region.
"""
