"""Defines `LBM_Config`, `ConfigLike`, and the `Float16C` conversion helpers.

`LBM_Config` carries the runtime toggles that parameterize an LBM run
(LES, DDF shifting, Float16C, boundary-condition set), and `Float16C`
provides the customized 16-bit floating-point format conversions described
in the FluidX3D reference.
"""
from .constants import Flags, _FlagSet
from std.collections import Set
from src.utils.custom_fp import Float16C

comptime fp32 = DType.float32
"""Alias for `DType.float32` used throughout the Float16C conversions."""
comptime uint16 = DType.uint16
"""Alias for `DType.uint16` used throughout the Float16C conversions."""


trait ConfigLike:
    """Declares the float-conversion API shared by LBM configurations.

    Not a binding trait until Mojo supports treating parameters as trait
    comptime members; it currently only declares the Float16C conversion
    helpers that conforming configs must provide.
    """

    # comptime DDF_shift:Bool
    # comptime LES:Bool
    # comptime KBC:Bool
    # comptime use_float16c:Bool
    # comptime f_dtype: Optional[DType]
    # comptime INCLUDED_BCs: Set[UInt8]
    # comptime second_moment:Bool
    @staticmethod
    @always_inline
    def fp32_to_fp16c(val: Scalar[fp32]) -> Scalar[uint16]:
        """Converts an `fp32` scalar to a Float16C `uint16` scalar.

        Args:
            val: The `fp32` value to convert.

        Returns:
            The Float16C-encoded value as a `uint16`.
        """
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32(val: UInt16) -> Scalar[fp32]:
        """Converts a Float16C `uint16` scalar back to an `fp32` scalar.

        Args:
            val: The Float16C-encoded `uint16` value.

        Returns:
            The decoded `fp32` value.
        """
        return Float16C.to_fp32(val)

    def set_f_dtype(self, float_dtype_for_math_ops: DType) -> DType:
        """Returns the `DType` to use for distribution-function math.

        Args:
            float_dtype_for_math_ops: The fallback float `DType` used when
                the config does not override the storage type.

        Returns:
            The `DType` to use for `f` math operations.
        """
        ...
        # return Self.f_dtype.value() if Self.f_dtype is not None else float_dtype


struct LBM_Config(ConfigLike):
    """Holds the runtime toggles that parameterize an LBM run.

    Records whether DDF shifting, Smagorinsky LES, KBC, and Float16C are
    enabled, the optional storage `DType` for the distribution function, and
    the set of boundary-condition flags the run expects to encounter. Each
    member is documented alongside its declaration below.
    """
    
    var DDF_shift: Bool
    """Whether DDF shifting is enabled for `f`."""
    var LES: Bool
    """Whether the Smagorinsky LES model is enabled."""
    var KBC: Bool
    """Reserved for the KBC collision model; currently always `False`."""
    var use_float16c: Bool
    """Whether `f` is stored as Float16C `uint16` values."""
    var f_dtype: Optional[DType]
    """Optional override `DType` for the distribution function."""
    var INCLUDED_BCs: Set[UInt8]
    """The set of boundary-condition flags valid for this run."""
    var second_moment: Bool
    """Whether the non-equilibrium second moment is computed each step."""
    var include_moving_boundary:Bool
    var collision_op:StaticString
    var valid_collision_ops:Set[StaticString] 
    
    def __init__(
        out self,
        *,
        collision_op:StaticString = 'SRT',
        LES: Bool = False,
        BCs: Set[UInt8] = {},
        DDF_shift: Bool = False,
        use_float16c: Bool = False,
        f_dtype: Optional[DType] = None,
        include_moving_boundary:Bool = False,
    ):
        """Constructs an `LBM_Config` from the supplied toggles.

        Enables the non-equilibrium second moment automatically when `LES`
        is `True`, forces `f_dtype` to `uint16` when `use_float16c` is
        `True`, and always includes `FLUID` and `SOLID` in the valid
        boundary-condition set.

        Args:
            LES: Whether to enable the Smagorinsky LES model (defaults to
                `False`).
            BCs: Set of uint8 containing Additional boundary-condition flags beyond `FLUID` and
                `SOLID` (defaults to the empty set).
            DDF_shift: Whether to enable DDF shifting (defaults to `False`).
            use_float16c: Whether to store `f` as Float16C `uint16`
                (defaults to `False`).
            f_dtype: Optional override `DType` for `f` (defaults to `None`).
        """
        
        self.collision_op = collision_op
        self.valid_collision_ops = {'SRT','TRT'}
        assert collision_op in self.valid_collision_ops

        self.DDF_shift = DDF_shift
        self.LES = LES
        self.KBC = False
        self.include_moving_boundary = include_moving_boundary
        self.second_moment = True if (LES) else False
        self.use_float16c = use_float16c
        self.f_dtype = DType.uint16 if use_float16c else f_dtype
        if self.use_float16c and self.f_dtype is not None:
            print(
                "WARNING: Float16c set to True and f_dtype was also specified."
                " Float16c overides the given dtype"
            )

        # Boundary Condition Check
        if len(BCs) == 0:
            self.INCLUDED_BCs = {Flags.FLUID, Flags.SOLID}
        else:
            __valid_bcs = materialize[_FlagSet]()
            if not BCs.issubset(__valid_bcs):
                print(
                    "Warning: Some Specified BC types are not standard: {}"
                    .format(BCs.difference(__valid_bcs))
                )
            # Ensure that fluid and solid nodes are always in the valid BC
            self.INCLUDED_BCs = {Flags.FLUID, Flags.SOLID}.union(BCs)

    def implies_f_noneq(self) -> Bool:
        """Returns `True` when the config requires the non-equilibrium part of `f`.

        Returns:
            `True` when LES is enabled, `False` otherwise.
        """
        return self.LES

    def set_f_dtype(self, float_dtype_for_math_ops: DType) -> DType:
        """Returns the `DType` to use for `f` math operations.

        Args:
            float_dtype_for_math_ops: The fallback float `DType` used when
                the config does not override the storage type.

        Returns:
            The configured `f` `DType` when set, otherwise the fallback.
        """
        return (
            self.f_dtype.value() if self.f_dtype
            is not None else float_dtype_for_math_ops
        )

    def collision_op_is_valid(self) -> Bool:
        return self.collision_op in self.valid_collision_ops

        

    @always_inline
    def enable_float16c(mut self):
        """Enables Float16C storage and sets `f_dtype` to `uint16`."""
        self.use_float16c = True
        self.f_dtype = DType.uint16

    def enable_DDF_shift(mut self):
        """Enables DDF shifting for improved numerical stability."""
        self.DDF_shift = True

    @staticmethod
    @always_inline
    def fp32_to_fp16c(val: Scalar[fp32]) -> Scalar[uint16]:
        """Converts an `fp32` scalar to a Float16C `uint16` scalar.

        Args:
            val: The `fp32` value to convert.

        Returns:
            The Float16C-encoded value as a `uint16`.
        """
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32[
        dtype: DType
    ](val: Scalar[dtype]) -> Scalar[fp32] where dtype == uint16:
        """Converts a Float16C `uint16` scalar back to an `fp32` scalar.

        Parameters:
            dtype: The `DType` of the input scalar; constrained to `uint16`.

        Args:
            val: The Float16C-encoded `uint16` value.

        Returns:
            The decoded `fp32` value.
        """
        return Float16C.to_fp32(val)

