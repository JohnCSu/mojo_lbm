"""Provides geometry primitives and immersed objects for the LBM domain.

The functions in this module mark fluid and solid nodes on the flag field by
embedding shapes such as spheres, circles, and boxes, and the `ImmersedObject`
struct collects the resulting fluid boundary nodes for use by force kernels.
"""
from .immersedObject import ImmersedObject
