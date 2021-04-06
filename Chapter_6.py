# Within a module, the module's name (as a string) is
# available as the value of the global variable __name__.
print(__name__)

import fibo

fib1 = fibo.fib
fib2 = fibo.fib2

fib1(100)
print(fib2(500))

fibo.fib(500)
print(fibo.__name__)

# Modules may contain executables that initialize the module.
# These are executed only the first time the module name is
# encountered in an import statement.

# Modules each have their own private symbol tables, which
# are used as the global symbol tables by all functions
# defined within the modules. Therefore there is no clash
# between module variables and user variables.

# Import names from the module directly into the importing
# module's symbol table. This does not introduce the module
# name into the local symbol table, so 'fibo' is not defined.
from fibo import fib, fib2
fib(1000)

# Import all names that a module defines. This imports all
# names except those beginning with an underscore. In
# practice this may not be used since it can import an
# unknown set of names. This notation is also less readable.
from fibo import *
print(fib2(1500))

import fibo as fibon
fibon.fib(50)

from fibo import fib as hello
hello(80)

# When the following is placed at the end of a module, it
# may be run as a script since 'python fibo.py <arguments>'
# executes the code in the module as if it was imported, but
# with __name__ == '__main__'. But then if the module is
# imported it does not run.
#if __name__ == '__main__':
#    import sys
#    fib(int(sys.argv[1]))

# When a module named spam is first imported, the interpreter first
# searches for a built-in module with that name. If not found
# it then searches for a file named spam.py in a list of
# directories given by the variable sys.path.

# The sys.path variable is a list of strings that determines
# the interpreter's search path for modules. It is
# initialized to a default path taken from the environment
# variable PYTHONPATH, or from a built-in default if
# PYTHONPATH is not set. It can be modified using standard
# list operations.
import sys
print(sys.path)
# sys.path.append()

# dir() lists all types of names: variables, modules,
# functions, etc. It does not list built-ins.
print(dir())
print(dir(fibo))
print(dir(sys))

import builtins
print(dir(builtins))

# Packages are a way of structuring Python's module
# namespace. A.B designates a submodule named B in a package
# named A. __init__.py files are required to make Python
# treat directories containing the file as packages. This
# prevents name collisions.

# If a package's __init__.py code defines a list named
# __all__, it is taken to be the list of module names that
# should be imported when from package import * is
# encountered. It is up to the package author to keep this
# list up-to-date.

# Leading dots on imports indicate the current and parent
# packages involved in a relative import.

# Packages have the special attribute __path__ that is
# initialized to be a list containing the name of the
# directory holding the package's __init__.py before the
# code in that file is executed. This variable can be
# modified.
