# utils
Basic building blocks methods. All methods should be fairly generic and have the following:
1. Not explicitley depend on grid or config  or any high abstraction struct. Parameters from these structs should be explicitely stated as their own independent parameters (e.g Q:Int rather than Q:Int = grid.Q)
2. Should always be functional and return a value. The exception are methods that store values
3. Be cpu/gpu generic and independent of mutability (Unless the method stores a value) and origin
4. Should be lbm_method generic i.e it can be used in Double buffer, esoteric or user own custom method
5. Should always have @always_inline/makes sense to have it
# ops
These build upon util functions to create more complicated lbm methods. These functions are most general.
Ops can be lbm method specific and do not have to return a value and it is prefered that no value is returned
1. Be cpu/gpu generic and independent of mutability (Unless the method stores a value)
2. No dependence on grid or config
3. LBM specific methods should live here 

# steps
These build on top of ops and utils to form methods that are called by the lbm kernel.
1. grid and config are parameters and any other needed parameters
2. Should be LBM-method independent and resolve any parameters needed for underyling ops and utils
3. Be cpu/gpu generic and independent of mutability (Unless the method stores a value) 

Utils -> Ops -> Steps