# Compile time Errors
Fix Errors first. Ignore warnings until all Errors are resolved first  

# Warnings
## implicit var declarations

### scoped statements

There are cases where a value of a variable is determined inside a scoped statement such as an if or for loop. In this case declare the variable outside the statement and then set the value inside of it:

```mojo
var c:Bool = True

var x:Int
if c:
    x = 1
else:
    x = 2

```

### Tuple declarations
tuples of implicitely copyable types can be done in the standard way:

``` mojo
var x = (1,2,3)

var a,b,c = x # Implicit Copy
```

But for tuples that are movable or copyable espcially if it is the output of , rather than copy use a ref to avoid uneeded copies:

``` mojo
var x = ([1,2,3],[1,2,3],[1,2,3])
ref a,b,c = x
```
### Documentation Warnings
Theses should be done last after all other warnings are resolved first. Here follow the info in @doctring-style-guide.md and style.md to analyse how documentation should be formatted and explained


