
@fieldwise_init
struct Unit[float_dtype:DType](ImplicitlyCopyable & Writable):
    comptime Float_Scalar = Scalar[Self.float_dtype]
    var physical:Self.Float_Scalar
    var lattice:Self.Float_Scalar

    def C_lat_to_phys(self) -> Self.Float_Scalar:
        return self.physical/self.lattice

    def C_phys_to_lat(self) -> Self.Float_Scalar:
        return self.lattice/self.physical
    

struct UnitSystem[float_dtype:DType](ImplicitlyCopyable & Writable):
    comptime Float_Scalar = Scalar[Self.float_dtype]
    comptime Unit_ = Unit[Self.float_dtype]
    var U:Self.Unit_
    var L:Self.Unit_
    var t:Self.Unit_
    var kinematic_viscosity:Self.Unit_
    var density:Self.Unit_ 
    var Re:Self.Float_Scalar
    var tau:Self.Float_Scalar

    var Mass:Self.Unit_
    var Force:Self.Unit_
    var Pressure:Self.Unit_
    def __init__(
        out self,
        u_physical:Self.Float_Scalar,
        u_lattice:Self.Float_Scalar,
        L_physical:Self.Float_Scalar,
        L_lattice:Self.Float_Scalar,
        kinematic_viscosity:Self.Float_Scalar,
        density:Self.Float_Scalar,
        ):
        self.U = Self.Unit_(u_physical,u_lattice)
        self.L = Self.Unit_(L_physical,L_lattice)
        self.t = Self.Unit_((L_physical/u_physical),L_lattice/u_lattice)

        self.density = Self.Unit_(density,1.)
        self.Re = (u_physical*L_physical)/kinematic_viscosity
        v_lat = u_lattice*L_lattice/self.Re
        self.kinematic_viscosity = Self.Unit_(kinematic_viscosity,v_lat)
        self.tau = v_lat/(1/3.) +0.5

        self.Mass = Self.Unit_(1.,self.density.C_phys_to_lat()*(self.L.C_phys_to_lat()**3)) # Unit Mass
        self.Force = Self.Unit_(1.,self.density.C_phys_to_lat()*self.L.C_phys_to_lat()**2*self.U.C_phys_to_lat()**2) # unit Force
        self.Pressure = Self.Unit_(1.,self.Force.C_phys_to_lat()/self.L.C_phys_to_lat()**2) # Unit Pressure
