"""Provides Python-interop helpers for loading the PyVista viewer module.

The functions here locate `pyvista_viewer.py` relative to this source file,
add its directory to the Python path, and construct a visualizer bound to an
`LBM_Grid`.
"""
from std.python import Python, PythonObject
from std.pathlib import Path
from std.os.path import dirname

from std.reflection import source_location, SourceLocation
from src.lbm import LBM_Grid


def pyvista_viewer_import() raises -> PythonObject:
    """Imports the `pyvista_viewer` Python module.

    Locates the directory of this source file at runtime via
    `source_location`, adds it to the Python path, and imports the
    `pyvista_viewer` module from it.

    Returns:
        The imported `pyvista_viewer` Python module object.
    """
    src = source_location()
    src_path = dirname(src.file_name())
    Python.add_to_path(src_path)  # Is this local to this file or to runtime?
    pyvista_viewer = Python.import_module("pyvista_viewer")
    return pyvista_viewer


def grid_viewer[
    grid: LBM_Grid
](subplot_shape: Tuple[Int, Int]) raises -> PythonObject:
    """Constructs a PyVista visualizer bound to an `LBM_Grid`.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain to visualize.

    Args:
        subplot_shape: The `(rows, cols)` shape of the subplot grid for the
            visualizer.

    Returns:
        A `Pyvista_Visualizer` instance configured for `grid`.

    Raises:
        Error: If either subplot dimension is less than 1.
    """
    pv_view = pyvista_viewer_import()

    if subplot_shape[0] < 1 or subplot_shape[1] < 1:
        raise Error("subplot shapes must be positive integers")

    visualizer = pv_view.Pyvista_Visualizer(
        grid.D,
        Python.tuple(grid.origin[0], grid.origin[1], grid.origin[2]),
        Python.tuple(
            grid.domain_size[0], grid.domain_size[1], grid.domain_size[2]
        ),
        Python.tuple(grid.shape[0], grid.shape[1], grid.shape[2]),
        Python.tuple(subplot_shape[0], subplot_shape[1]),
    )

    return visualizer
