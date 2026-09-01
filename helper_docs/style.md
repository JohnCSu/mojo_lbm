# Inplace or output return type but not both

Functions defined should either be inplace or functional and output a new variable. No mixing of both inplace ops and having a return type. Exception are struct methods which may mutate self and return something. 

# Inplace Functions

## Max mutations
Inplace functions should only at most mutate 3 args the rest should be immutabel. Ideally only 1 arg be mutated. Exception are gpu kernels and cpu entry point functions.

## order of args
mutable args should always be placed as the first args followed by immutable args


# Function style
This is a guideline not a strict rule. Theses shouldn't be changed but recorded as not following the guideline
- use the prefix get for small functions that return a type e.g. get_density
- use the prefix set for small functions that mutate an input e.g. set_adjacent_flags
- functions and method calls use "_" between words and lowercase (unless word is an acronym) e.g get_density
- Structs and traits Prefer the use of CamelCase