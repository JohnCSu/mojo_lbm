comptime FLUID_NODE:Scalar[DType.uint8] = 0
comptime SOLID_NODE:Scalar[DType.uint8] = 1

comptime cs = (1./3.)**0.5
comptime cs_squared = (1./3.)


struct Flags():
    comptime FLUID:UInt8 =0
    comptime SOLID:UInt8 = 1
    comptime EQUILIBRIUM:UInt8 = 2


comptime _FlagSet = {Flags.FLUID,Flags.SOLID,Flags.EQUILIBRIUM}
