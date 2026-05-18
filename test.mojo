# from std.builtin.variadics import 
from std.sys.intrinsics import _type_is_eq
from std.testing import assert_equal

# Create a type list
comptime tl = TypeList[Trait=AnyType, Int, String, Float64]()

def main():
    # Query size
    assert_equal(tl.size, 3)

    # Check membership
    comptime assert tl.contains[Int]
    comptime assert not tl.contains[Bool]

    # Index into the list
    comptime assert _type_is_eq[tl[0], Int]()