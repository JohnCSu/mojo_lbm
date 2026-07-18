"""Provides visualization helpers backed by Python modules.

Exposes convenience functions that import a PyVista-based viewer from Python
and construct a visualizer bound to an `LBM_Grid` instance.
"""
from ._python_importer import pyvista_viewer_import, grid_viewer
