from std.builtin.device_passable import DevicePassable

struct RuntimeParams[float_dtype:DType](DevicePassable & ImplicitlyCopyable):
    comptime device_type:AnyType = Self
    comptime Float = Scalar[Self.float_dtype]
    var Cs:Self.Float
    var TRT_magic_param:Self.Float

    def __init__(
        out self,
        *,
        Cs:Self.Float = 0.1,
        TRT_magic_param:Self.Float = 3./16.
    ):
        self.Cs = Cs
        self.TRT_magic_param = TRT_magic_param

    @staticmethod
    def get_type_name() -> String:
        return String(
            "RuntimeParams[",
            reflect[type_of(Self.float_dtype)]().name(),
            "]")
   
    def _to_device_type(
        self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self.copy()

    def tau_asymm(self,tau:Self.Float) -> Self.Float:
        return 0.5 + self.TRT_magic_param/(tau-0.5)
