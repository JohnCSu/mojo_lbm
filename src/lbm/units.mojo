"""Defines `Unit` and `UnitSystem` for converting between physical and lattice units.

`Unit` pairs a physical value with its lattice counterpart and exposes the
conversion factors between them. `UnitSystem` builds a full set of LBM unit
conversions (tau, Reynolds number, time step, mass, force, pressure) from a
small set of physical inputs.
"""


@fieldwise_init
struct Unit[float_dtype: DType](ImplicitlyCopyable & Writable):
    """Pairs a physical value with its lattice counterpart.

    Stores both representations so the conversion factors between physical
    and lattice units can be derived on demand.

    Parameters:
        float_dtype: The `DType` used for both values.
    """

    comptime Float_Scalar = Scalar[Self.float_dtype]
    var physical: Self.Float_Scalar
    """The value in physical units."""
    var lattice: Self.Float_Scalar
    """The value in lattice units."""

    @always_inline
    def C_lat_to_phys(self) -> Self.Float_Scalar:
        """Returns the conversion factor from lattice to physical units.

        Returns:
            The factor `physical / lattice`.
        """
        return self.physical / self.lattice

    @always_inline
    def C_phys_to_lat(self) -> Self.Float_Scalar:
        """Returns the conversion factor from physical to lattice units.

        Returns:
            The factor `lattice / physical`.
        """
        return self.lattice / self.physical


struct UnitSystem[float_dtype: DType, D: Int](ImplicitlyCopyable & Writable):
    """Stores the unit conversions for an LBM system.

    Holds the velocity, length, time, viscosity, and density `Unit`s plus the
    derived Reynolds number, relaxation time `tau`, and time step `dt`, and
    computes the unit mass, force, and pressure for the configured dimension.

    Following general LBM conventions, lattice length, time, and density are
    set to 1, and the unit mass, force, and pressure are defined so their
    lattice counterparts are also 1. The caller is responsible for keeping
    the physical units self-consistent (for example, N-mm-s or N-m-s); no
    consistency checking is performed.

    Parameters:
        float_dtype: The `DType` used for all stored values.
        D: The spatial dimension of the simulation.
    """

    comptime Float_Scalar = Scalar[Self.float_dtype]
    comptime Unit_ = Unit[Self.float_dtype]
    var U: Self.Unit_
    """The velocity unit pair."""
    var L: Self.Unit_
    """The length unit pair."""
    var t: Self.Unit_
    """The time unit pair."""
    var kinematic_viscosity: Self.Unit_
    """The kinematic viscosity unit pair."""
    var density: Self.Unit_
    """The density unit pair."""

    var Re: Self.Float_Scalar
    """The Reynolds number of the simulation."""
    var tau: Self.Float_Scalar
    """The SRT relaxation time in lattice units."""
    var dt: Self.Float_Scalar
    """The physical time step per lattice step."""

    var mass: Self.Unit_
    """The unit mass for the configured dimension."""
    var force: Self.Unit_
    """The unit force for the configured dimension."""
    var pressure: Self.Unit_
    """The unit pressure for the configured dimension."""

    def __init__(
        out self,
        u_physical: Self.Float_Scalar,
        u_lattice: Self.Float_Scalar,
        L_physical: Self.Float_Scalar,
        L_lattice: Self.Float_Scalar,
        density: Self.Float_Scalar,  # M/L**3
        kinematic_viscosity: Self.Float_Scalar,  # L/T**2
    ):
        """Constructs a `UnitSystem` from a kinematic viscosity.

        Args:
            u_physical: The physical velocity scale, typically the free
                stream or inlet velocity.
            u_lattice: The lattice velocity. Standard range is `[0.001, 0.1]`.
            L_physical: The physical length scale.
            L_lattice: The lattice length scale, approximately the number of
                lattice points along a direction that represents
                `L_physical`.
            density: The density of the actual fluid. Does not influence the
                flow physics; used for unit conversion only.
            kinematic_viscosity: The kinematic viscosity, equivalent to
                `dynamic_viscosity / density`.
        """
        self.U = Self.Unit_(u_physical, u_lattice)
        self.L = Self.Unit_(L_physical, L_lattice)
        self.t = Self.Unit_(
            (L_physical / u_physical) / (L_lattice / u_lattice), 1.0
        )
        self.density = Self.Unit_(density, 1.0)

        self.Re = (self.U.physical * self.L.physical) / kinematic_viscosity
        v_lat = self.U.lattice * self.L.lattice / self.Re

        self.dt = self.t.physical
        self.tau = v_lat / (1 / 3.0) + 0.5

        self.kinematic_viscosity = Self.Unit_(kinematic_viscosity, v_lat)

        # Useful for analysing drag and mass flow
        self.Mass = Self.Unit_(
            self.density.C_lat_to_phys() * (self.L.C_lat_to_phys() ** Self.D),
            1.0,
        )
        self.Force = Self.Unit_(
            self.density.C_lat_to_phys()
            * self.L.C_lat_to_phys() ** (Self.D - 1)
            * self.U.C_lat_to_phys() ** 2,
            1.0,
        )  # unit Force
        self.Pressure = Self.Unit_(
            self.Force.C_lat_to_phys() / self.L.C_lat_to_phys() ** (Self.D - 1),
            1.0,
        )  # Unit Pressure

    def __init__(
        out self,
        u_physical: Self.Float_Scalar,
        u_lattice: Self.Float_Scalar,
        L_physical: Self.Float_Scalar,
        L_lattice: Self.Float_Scalar,
        density: Self.Float_Scalar,
        *,
        dynamic_viscosity: Self.Float_Scalar,
    ):
        """Constructs a `UnitSystem` from a dynamic viscosity.

        Derives the kinematic viscosity as `dynamic_viscosity / density` and
        delegates to the kinematic-viscosity overload.

        Args:
            u_physical: The physical velocity scale.
            u_lattice: The lattice velocity.
            L_physical: The physical length scale.
            L_lattice: The lattice length scale.
            density: The density of the actual fluid.
            dynamic_viscosity: The dynamic viscosity of the fluid.
        """
        kinematic_viscosity = dynamic_viscosity / density
        self = Self(
            u_physical,
            u_lattice,
            L_physical,
            L_lattice,
            density,
            kinematic_viscosity,
        )

    def __init__(
        out self,
        u_physical: Self.Float_Scalar,
        u_lattice: Self.Float_Scalar,
        L_physical: Self.Float_Scalar,
        L_lattice: Self.Float_Scalar,
        density: Self.Float_Scalar,
        *,
        Re: Self.Float_Scalar,
    ):
        """Constructs a `UnitSystem` from a target Reynolds number.

        Derives the kinematic viscosity as `u_physical * L_physical / Re` and
        delegates to the kinematic-viscosity overload.

        Args:
            u_physical: The physical velocity scale.
            u_lattice: The lattice velocity.
            L_physical: The physical length scale.
            L_lattice: The lattice length scale.
            density: The density of the actual fluid.
            Re: The target Reynolds number.
        """
        kinematic_viscosity = u_physical * L_physical / Re
        self = Self(
            u_physical,
            u_lattice,
            L_physical,
            L_lattice,
            density,
            kinematic_viscosity,
        )
