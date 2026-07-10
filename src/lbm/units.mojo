
@fieldwise_init
struct Unit[float_dtype:DType](ImplicitlyCopyable & Writable):
    comptime Float_Scalar = Scalar[Self.float_dtype]
    var physical:Self.Float_Scalar
    var lattice:Self.Float_Scalar

    @always_inline
    def C_lat_to_phys(self) -> Self.Float_Scalar:
        """
        Conversion factor to convert from lattice units to physical units.
        """
        return self.physical/self.lattice

    @always_inline
    def C_phys_to_lat(self) -> Self.Float_Scalar:
        """
        Conversion factor to convert from physical units to lattice units 
        """
        return self.lattice/self.physical
    

struct UnitSystem[float_dtype:DType,D:Int](ImplicitlyCopyable & Writable):
    """
    Struct to store relevant unit conversion values for LBM System.

    Automatically Calculates:

        - tau (Float) 
        - Re (Float) 
        - dt (Float)
        - Mass (Unit)
        - Force (Unit)
        - Pressure (Unit)

    Following general LBM conventions, lattice length,time and density are set to 1. 
    Mass, Force and Pressure are also defined such that their lattice counterpart is set to 1. 

    It is up to the user to ensure that the physical units are consistent 
    with each other (e.g. use N - mm - s or N- m - s). No checking is performed.
    """
    comptime Float_Scalar = Scalar[Self.float_dtype]
    comptime Unit_ = Unit[Self.float_dtype]
    var U:Self.Unit_
    var L:Self.Unit_
    var t:Self.Unit_
    var kinematic_viscosity:Self.Unit_
    var density:Self.Unit_ 
     
    var Re:Self.Float_Scalar
    var tau:Self.Float_Scalar
    var dt:Self.Float_Scalar

    var Mass:Self.Unit_
    var Force:Self.Unit_
    var Pressure:Self.Unit_
    
    def __init__(
        out self,
        u_physical:Self.Float_Scalar,
        u_lattice:Self.Float_Scalar,
        L_physical:Self.Float_Scalar,
        L_lattice:Self.Float_Scalar,
        density:Self.Float_Scalar, # M/L**3
        kinematic_viscosity:Self.Float_Scalar, # L/T**2
        ):
        """
        Create the Unit System fot the LBM model.

        Args:
            u_physical: Physical U scale. typically one chooses the free stream velocity/inlet velocity.
            u_lattice: Lattice velocity. User is free to choose this for LBM. Standard range is [0.001,0.1].
            L_physical: Physical Length Scale.
            L_lattice: Lattice Length Scale. This should be the approximate number 
                of lattice points along a direction that represents L_physical in the grid.
            density: Density of actual fluid. Does not influence the physics of the flow.
                Used for Unit conversion only.
            kinematic_viscosity: Equivalent to dynamic_viscosity/density.
        """
        
        self.U = Self.Unit_(u_physical,u_lattice)
        self.L = Self.Unit_(L_physical,L_lattice)
        self.t = Self.Unit_((L_physical/u_physical)/(L_lattice/u_lattice),1.)
        self.density = Self.Unit_(density,1.)
        
        self.Re = (self.U.physical*self.L.physical)/kinematic_viscosity
        v_lat = self.U.lattice*self.L.lattice/self.Re

        self.dt = self.t.physical
        self.tau = v_lat/(1/3.) +0.5

        self.kinematic_viscosity = Self.Unit_(kinematic_viscosity,v_lat)
        
        # Useful for analysing drag and mass flow
        self.Mass = Self.Unit_(self.density.C_lat_to_phys() * (self.L.C_lat_to_phys()**Self.D) , 1. )
        self.Force = Self.Unit_(self.density.C_lat_to_phys() * self.L.C_lat_to_phys()**(Self.D-1) * self.U.C_lat_to_phys()**2,1.) # unit Force
        self.Pressure = Self.Unit_(self.Force.C_lat_to_phys()/self.L.C_lat_to_phys()**(Self.D-1),1.) # Unit Pressure


    def __init__(
        out self,
        u_physical:Self.Float_Scalar,
        u_lattice:Self.Float_Scalar,
        L_physical:Self.Float_Scalar,
        L_lattice:Self.Float_Scalar,
        density:Self.Float_Scalar,
        *,
        dynamic_viscosity:Self.Float_Scalar,
        ):
        kinematic_viscosity = dynamic_viscosity/density
        self = Self(u_physical,u_lattice,L_physical,L_lattice,density,kinematic_viscosity)

    def __init__(
        out self,
        u_physical:Self.Float_Scalar,
        u_lattice:Self.Float_Scalar,
        L_physical:Self.Float_Scalar,
        L_lattice:Self.Float_Scalar,
        density:Self.Float_Scalar,
        *,
        Re:Self.Float_Scalar,
        ):
        kinematic_viscosity = u_physical*L_physical/Re
        self = Self(u_physical,u_lattice,L_physical,L_lattice,density,kinematic_viscosity)