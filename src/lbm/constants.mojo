"""Defines compile-time constants shared across the LBM solver.

Holds the flag values used to label lattice nodes (fluid, solid, equilibrium),
the lattice speed of sound `cs`, and the set of valid boundary-condition
flags consumed by `LBM_Config`.
"""


comptime cs = (1.0 / 3.0) ** 0.5
"""The lattice speed of sound, $$\\sqrt{1/3}$$."""
comptime cs_squared = (1.0 / 3.0)
"""The square of the lattice speed of sound, $$1/3$$."""


struct Flags:
    """Collects the boundary-condition flag values as compile-time constants.

    These flag values label lattice nodes for the streaming and collision
    kernels: `FLUID` for interior nodes, `SOLID` for wall nodes, and
    `EQUILIBRIUM` for nodes that should be reset to the equilibrium
    distribution each step.
    """

    comptime FLUID: UInt8 = 0
    """Flag value for a fluid node."""
    comptime SOLID: UInt8 = 1
    """Flag value for a solid (wall) node."""
    comptime EQUILIBRIUM: UInt8 = 2
    """Flag value for an equilibrium boundary node."""


comptime _FlagSet = {Flags.FLUID, Flags.SOLID, Flags.EQUILIBRIUM}
"""The set of valid boundary-condition flags accepted by `LBM_Config`."""


comptime FLUID_NODE: Scalar[DType.uint8] = Flags.FLUID
"""Flag value for a fluid node."""
comptime SOLID_NODE: Scalar[DType.uint8] = Flags.SOLID
"""Flag value for a solid (wall) node."""
