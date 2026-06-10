from std.algorithm.functional import vectorize
from std.sys import simd_width_of
comptime simd_width = 4
comptime test = simd_width_of[DType.float32]()

def main() raises:
    f_new:InlineArray[Scalar[DType.float32],9] = [1,2,3,4,5,6,7,8,9]
    print(test)
    rho:Float32 = 0

    @always_inline
    def vector_sum[width:Int](i:Int) {read f_new, mut rho}:
        f_ptr = f_new.unsafe_ptr()
        print('i= ',i)
        if i  <= len(f_new): # Ensure the load is within bounds
            rho += f_ptr.load[width](i).reduce_add()
            print(f_ptr.load[width](i))
    print(rho)
    vectorize[simd_width](len(f_new),vector_sum)

    print(rho)

