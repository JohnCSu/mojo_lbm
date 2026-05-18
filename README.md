# MOJO_SIM

A basic implementation for 2D D2Q9 LBM for Mojo. This is a learning exercise.

Currently at the stage of creating a simple Row Major LBM

# Timeline
- [] Create function to set BC - Moving and No Slip
- [] Create LBM kernel with mid grid bounceback



# Reflection
- 2026/05/12
    - Awkward slicing syntax
    - Type System can be annoying
    - Int and Int32 for Gpu kernels type mismatching
    - Lack of clarity what can be passed to GPU
    - Very Barebones so have to basically build everything from scratch
    - Maybe to low level for now to incentivise a switch from CUDA or Python DSLs

- 2026_05/14
    - Optional is weird and doesnt make sense
    - if self.last_used_cpu is False: -> Bool dont have __is__ implemented??