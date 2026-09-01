from std.math import sqrt, abs, min, max




trait SDF(Copyable):
    comptime float_dtype:DType
    def __call__(self,point:Array[Scalar[Self.float_dtype],3]) -> Scalar[Self.float_dtype]:
        ...

    def bounding_box(self) -> Tuple[Array[Scalar[Self.float_dtype],3],Array[Scalar[Self.float_dtype],3]]:
        '''
        Return the min_point and max point AABB of the sdf
        '''
        ...

    def is_inside(self,point:Array[Scalar[Self.float_dtype],3]) -> Bool:
        return self(point) < 0
    
    def is_outside(self,point:Array[Scalar[Self.float_dtype],3]) -> Bool:
        return self(point) > 0
    
    
# @fieldwise_init
struct Box[float_dtype_:DType](SDF):
    comptime float_dtype = Self.float_dtype_
    var min_point:Array[Scalar[Self.float_dtype_],3]
    var max_point:Array[Scalar[Self.float_dtype_],3]

    var center:Array[Scalar[Self.float_dtype_],3]
    var half_dimensions:Array[Scalar[Self.float_dtype_],3]

    def __init__(out self,min_point:Array[Scalar[Self.float_dtype_],3],max_point:Array[Scalar[Self.float_dtype_],3]):
        self.max_point = max_point.copy()
        self.min_point = min_point.copy()
       
        self.center = max_point.copy()
        self.half_dimensions = type_of(self.center)(uninitialized = True)
        comptime for i in range(3):
            self.half_dimensions[i] = abs(max_point[i] - min_point[i])/2
            self.center[i] = (self.max_point[i] + self.min_point[i])/2

    def __call__(self,point:Array[Scalar[Self.float_dtype],3])-> Scalar[Self.float_dtype]:
        var q = type_of(point)(uninitialized = True)
        
        comptime for i in range(3):
            q[i] = abs(point[i] - self.center[i]) - self.half_dimensions[i]

        var outside_sq:Scalar[Self.float_dtype] = 0.
        var inside = q[0]
        comptime for i in range(3):
            var clamped = max(q[i],0.)
            outside_sq += clamped*clamped
            inside = max(inside,q[i])
        return sqrt(outside_sq) + min(inside,0.)

    def bounding_box(self) -> Tuple[Array[Scalar[Self.float_dtype],3],Array[Scalar[Self.float_dtype],3]]:
        return self.min_point.copy(),self.max_point.copy()

    def width(self) -> Scalar[Self.float_dtype_]:
        return self.half_dimensions[0]*2

    def length(self) -> Scalar[Self.float_dtype_]:
        return self.half_dimensions[1]*2

    def depth(self) -> Scalar[Self.float_dtype_]:
        return self.half_dimensions[2]*2

@fieldwise_init
struct Sphere[float_dtype_:DType](SDF):
    comptime float_dtype = Self.float_dtype_
    var center:Array[Scalar[Self.float_dtype_],3]
    var radius:Scalar[Self.float_dtype_]

    def __call__(self,point:Array[Scalar[Self.float_dtype],3])-> Scalar[Self.float_dtype]:
        var sq_mag:Scalar[Self.float_dtype] = 0.
        comptime for i in range(3):
            var point_shift = point[i] - self.center[i]
            sq_mag += point_shift*point_shift
        return sqrt(sq_mag)-self.radius

    def bounding_box(self) -> Tuple[Array[Scalar[Self.float_dtype],3],Array[Scalar[Self.float_dtype],3]]:
        var min_point = self.center.copy()
        var max_point = self.center.copy()
        comptime for i in range(3):
            min_point[i] -= self.radius
            max_point[i] += self.radius
        return min_point^,max_point^


struct Cylinder[float_dtype_:DType,axis:Int = 2](SDF):
    '''Axis-aligned capped cylinder along the comptime `axis` (0=x, 1=y, 2=z).

    Solid for `dist(p, axis_line) <= radius` and `|axis coordinate - center| <= half_height`.
    '''
    comptime float_dtype = Self.float_dtype_
    var center:Array[Scalar[Self.float_dtype_],3]
    var radius:Scalar[Self.float_dtype_]
    var half_height:Scalar[Self.float_dtype_]


    def __init__(out self,center:Array[Scalar[Self.float_dtype_],3],radius:Scalar[Self.float_dtype_],half_height:Scalar[Self.float_dtype_]):
        self.center = center.copy()
        self.radius = radius
        self.half_height = half_height


    def __call__(self,point:Array[Scalar[Self.float_dtype],3])-> Scalar[Self.float_dtype]:
        var radial_sq:Scalar[Self.float_dtype] = 0.
        var axial:Scalar[Self.float_dtype] = 0.
        comptime for i in range(3):
            var d = point[i] - self.center[i]
            if i == Self.axis:
                axial = d
            else:
                radial_sq += d*d
        var radial = sqrt(radial_sq)

        var d_r = radial - self.radius
        var d_h = abs(axial) - self.half_height
        var outside_r = max(d_r,0.)
        var outside_h = max(d_h,0.)
        return min(max(d_r,d_h),0.) + sqrt(outside_r*outside_r + outside_h*outside_h)


    def bounding_box(self) -> Tuple[Array[Scalar[Self.float_dtype],3],Array[Scalar[Self.float_dtype],3]]:
        var min_point = self.center.copy()
        var max_point = self.center.copy()
        comptime for i in range(3):
            var extent = self.radius
            if i == Self.axis:
                extent = self.half_height
            min_point[i] -= extent
            max_point[i] += extent
        return min_point^,max_point^
