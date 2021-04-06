# Clases.

# Classes provide a means of bundling data and functionality together. Creating a new
# class creates a new type of object, allowing new instances of that type to be made.
# Each class can have attributes attached to it. Class instances can also have methods
# defined by the class for modifying its state.

# Passing an object can be cheap since (if?) only a pointer is passed by the
# implementation.

# Python Scopes and Namespaces.
# A namespace is a mapping from names to objects. Most namespaces are currently
# implemented as Python dictionaries. There is no relation between names in different
# namespaces.

# An attribute is any name following a dot. Module attributes are writeable. For
# example with modname.ingredient = 'rice'.

# Namespaces are created at different moments and have different lifetimes. The
# namespace containing built-in functions is created when the Python interpreter
# starts up. The local namespace for a function is created when the function is
# called, and deleted when the function returns or raises an unhandled exception.
# Recursive invocations each have their own local namespace.

# A scope is a textual region of a Python program where a namespace is directly
# accessible, meaning that an unqualified reference to a name attempts to find the
# name in the namespace. At any time during execution there are 3 or 4 nested scopes
# who namespaces are directly accessible.
# 1) The innermost scope, searched first, containing local names.
# 2) The scopes of enclosing functions, searched starting with the nearest enclosing
# scope, containing non-local but also non-global names.
# 3) The scope containing the module's global names.
# 4) The scope which is the namespace containing built-in names.

# Demonstrate the local, nonlocal, and global scopes.
def scope_test():
    """Print a variable defined in each namespace."""

    def do_local():
        spam = 'local spam'

    def do_nonlocal():
        nonlocal spam
        spam = 'nonlocal spam'

    def do_global():
        global spam
        spam = 'global spam'

    spam = 'test spam'
    do_local()
    print('After local assignment:', spam)
    do_nonlocal()
    print('After nonlocal assignment:', spam)
    do_global()
    print('After global assignment:', spam)

scope_test()
print('In global scope:', spam)

# Class definitions create a new local namespace. Functions defined there bind the
# name of the new function there.

# When a class definition is left normally, a 'class object' is created, which is
# basically a wrapper around the contents of the namespace created by the class
# definition.

# Class objects support two kinds of operations:
# 1) Attribute references.
# 2) Instantiation.
# Valid attribute names are all the names that were in the class's namespace when the
# class object was created. In the following, the 'i' and 'f' are in the namespace,
# and so MyClass.i and MyClass.f are valid attribute references.
class MyClass:
    """Demonstrating the class object namespace."""

    i = 12345

    def f(self):
        return 'Hello, world!'

print(MyClass.i)
greeting = MyClass.f(1)
print(greeting)

# Instantiate a class.
x = MyClass()
print(x.i)

# Classes can be initialized in a particular state using
#def __init__(self):
#    self.data = []
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

z = Complex(1, 4)
print(z.r, z.i)

# Data attributes do not need to be declared. Like local variables, they come into
# existence when they are assigned.
z.bird = 'Parrot'
print(z.bird)

# Apparently MyClass.f is not the same as x.f.
print(x.f() == MyClass.f(x))

class Dog:
    """Dogs are all canines, but they all have different names and tricks."""

    kind = 'canine'

    def __init__(self, name):
        self.name = name
        self.tricks = [] # Dogs don't learn their tricks at birth.

    def add_trick(self, trick):
        self.tricks.append(trick)

d1 = Dog('Billy')
d2 = Dog('Bobbo')
d1.add_trick('Roll over.')
d2.add_trick('Cook dinner.')

print(d1.name, d2.name, d1.tricks, d2.tricks)

print(Dog.__doc__)

# Inheritance.

# When resolving attribute references, the search is done first in the class, then
# in the base class. This procedure continues recursively if the base class itself
# is derived.

# Instantiation works the same for derived classes. Method references are resolved
# according to the previous paragraph. First check the class, then recursively
# through the other base classes.

# Derived classes may override methods of their base classes.
print(type(Dog))

# Multiple inheritance is possible by defining a class with multiple base classes.
# Name wrangling with preceding underscores and possibly a trailing underscore can
# be used to avoid name clashes.

# Empty classes can be used for bundling data.
class Customer:
    pass

cust1 = Customer()

cust1.name = 'Philip'
cust1.money = 'Millions'
cust1.location = 'New York'

print(dir(cust1))

# next() for iterables.
s = 'abc'
it = iter(s)
print(it)
print(next(it), next(it), next(it))

# 'Yield' for generators.

# Generators can be coded with a similar syntax to list comprehensions, but with
# parentheses instead of square brackets. They tend to be more memory friendly than
# equivalent list comprehensions.
print(sum(x*x for x in range(11)))
xvec = [1, 2, 3]
yvec = [4, 5, 6]
print(sum(x*y for x,y in zip(xvec, yvec))) # Dot product.
