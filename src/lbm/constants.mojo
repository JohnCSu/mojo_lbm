"""Defines compile-time constants shared by all LBM modules.

Collects boundary-condition flag values, the speed of sound, collision
operator names, and LBM streaming-method identifiers as comptime aliases so
they can be referenced by the solver, preprocessor, and GPU kernels at
compile time.
"""
from std.collections import Set

comptime cs = (1.0 / 3.0) ** 0.5
"""The lattice speed of sound, $$\\sqrt{1/3}$$."""
comptime cs_squared = (1.0 / 3.0)
"""The square of the lattice speed of sound, $$1/3$$."""


trait Enum(ImplicitlyCopyable & Hashable & Equatable):
    ...

@fieldwise_init
struct LBM_method(Enum):
    """Collects the LBM streaming-method names as compile-time constants.

    The valid methods are `DOUBLE_BUFFER` and `ESOTERIC_PULL`. The
    `valid_set` alias is used by `LBM_Config` to reject unknown methods at
    compile time.
    """
    var _value:Int
    comptime DOUBLE_BUFFER = LBM_method(0)
    comptime ESOTERIC_PULL = LBM_method(1)
    comptime MOMENT_REPRESENTATION = LBM_method(2)
    comptime valid_set:Set[LBM_method] = {Self.DOUBLE_BUFFER,Self.ESOTERIC_PULL}



comptime DOUBLE_BUFFER = LBM_method.DOUBLE_BUFFER
"""Convenience alias for `LBM_method.DOUBLE_BUFFER`."""
comptime ESOTERIC_PULL = LBM_method.ESOTERIC_PULL
"""Convenience alias for `LBM_method.ESOTERIC_PULL`."""
# comptime lbm_methods:Set[StaticString] = {DOUBLE_BUFFER,ESOTERIC_PULL}
    
struct Collisions:
    """Collects the collision-operator names as compile-time constants.

    The valid operators are `SRT`, `TRT`, `KBC`, and `RLBM`.
    `that_need_fneq` lists the operators that require the non-equilibrium
    part of the distribution function.
    """

    comptime SRT:StaticString = 'SRT'
    comptime TRT:StaticString = 'TRT'
    comptime KBC:StaticString = 'KBC'
    comptime RLBM:StaticString = 'RLBM'
    comptime valid_set:Set[StaticString] = {Self.SRT,Self.TRT,Self.RLBM,Self.KBC}

    comptime that_need_fneq:Set[StaticString] = {Self.KBC,Self.RLBM}


struct Flags:
    """Collects the boundary-condition flag values as compile-time constants.

    These flag values label lattice nodes for the streaming and collision
    kernels: `FLUID` for interior nodes, `SOLID` for wall nodes, and
    `EQUILIBRIUM` for nodes that should be reset to the equilibrium
    distribution each step. `FLUID_BOUNDARY` reuses the value `2` to tag fluid
    nodes that are adjacent to a solid — for now it shares the value with
    `EQUILIBRIUM`.
    """

    comptime FLUID: UInt8 = 0
    """Flag value for a fluid node."""
    comptime SOLID: UInt8 = 1
    """Flag value for a solid (wall) node."""
    comptime EQUILIBRIUM: UInt8 = 2
    """Flag value for an equilibrium boundary node."""
    comptime FLUID_BOUNDARY: UInt8 = 2
    """Flag value for a fluid node adjacent to a solid (shares value with
    `EQUILIBRIUM` for now)."""




@fieldwise_init
struct Bounceback_method(Enum):
    var _value:Int
    comptime MID_GRID = Bounceback_method(0) 
    comptime BOUZIDI = Bounceback_method(1)


comptime _FlagSet = {Flags.FLUID, Flags.SOLID, Flags.EQUILIBRIUM}
"""The set of valid boundary-condition flags accepted by `LBM_Config`."""


comptime FLUID_NODE: Scalar[DType.uint8] = Flags.FLUID
"""Flag value for a fluid node."""
comptime SOLID_NODE: Scalar[DType.uint8] = Flags.SOLID
"""Flag value for a solid (wall) node."""
comptime FLUID_BOUNDARY_NODE: Scalar[DType.uint8] = Flags.FLUID_BOUNDARY
"""Flag value for a fluid node adjacent to a solid."""


