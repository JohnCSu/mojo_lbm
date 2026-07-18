"""Provides a GPU-accelerated Lattice Boltzmann Method (LBM) solver.

The solver implements the Single Relaxation Time (SRT) model for
incompressible flow and supports the D2Q9, D3Q19, and D3Q27 lattice models
through Mojo metaprogramming. A single kernel serves all dimensions and
lattice models, and the layout-independent implementation works for row-major,
column-major, and tiled arrays using natural `(x, y, z, q)` indexing.

Reach for this package to run LBM simulations on Nvidia, AMD, or Apple GPUs
using only the Mojo standard library.
"""
