from std.collections import Set

comptime cs = (1.0 / 3.0) ** 0.5
"""The lattice speed of sound, $$\\sqrt{1/3}$$."""
comptime cs_squared = (1.0 / 3.0)
"""The square of the lattice speed of sound, $$1/3$$."""




struct Lbm_methods:
    comptime DOUBLE_BUFFER:StaticString = 'double buffer'
    comptime ESOTERIC_PULL:StaticString = 'esoteric pull'
    comptime MOMENT_REPRESENTATION:StaticString = 'moment representation'
    comptime valid_set:Set[StaticString] = {Self.DOUBLE_BUFFER,Self.ESOTERIC_PULL}

comptime DOUBLE_BUFFER:StaticString = Lbm_methods.DOUBLE_BUFFER
comptime ESOTERIC_PULL:StaticString = Lbm_methods.ESOTERIC_PULL
# comptime lbm_methods:Set[StaticString] = {DOUBLE_BUFFER,ESOTERIC_PULL}
    
struct Collisions:
    comptime SRT:StaticString = 'SRT'
    comptime TRT:StaticString = 'TRT'
    comptime KBC:StaticString = 'KBC'
    comptime RLBM:StaticString = 'RLBM'
    comptime valid_set:Set[StaticString] = {Self.SRT,Self.TRT,Self.RLBM}

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
