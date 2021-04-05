# Leftover from Chapter 3: Empty lists and 0's are considered to be False. Any
# non-empty list or positive (?) number is True.

# if, elif, and else.

chickens = 'raw'
for i in range(3):
    if chickens == 'raw':
        print('Cook that chicken!')
        chickens = 'cooked'
    elif chickens == 'cooked':
        print('All good.')
        chickens = 'burnt'
    else:
        print('Uh-oh, it\'s burnt. Now what do we do?')

# Python's 'for' statement iterates over the items of any
# sequence.

ingredients = ['chicken', 'pork', 'mash']
for i in ingredients:
    print(i, len(i))

# The range function in Python can be used like the Table
# indices in Mathematica by including start and end points
# with the step. Although the increment apparently needs to
# be a legal index.

print(list(range(1,5,2)))
print(list(range(-10,-20,-1)))
print(list(range(-10,-20,1)))

# The range() function does not print directly. It is not a
# list, rather it is an iterable. That means it can be passed
# to functions and constructs that expect something from which
# they can obtain a sequence of items. The for statement is
# one example. Another is sum().

print(sum(range(1,5)))

# The 'break' statement breaks out of the innermost enclosing
# for or while loop.

# Loops may have else clauses that are executed when the loop
# iterable is exhausted (for) or when the condition becomes
# False (while) but not when the loop is broken by a break
# statement.

# The continue statement continues with the next iteration of
# the loop.

# The pass statement does nothing, and can be used as a
# placeholder when a statement is required syntactically but
# the program requires no action.

# Turn the Fibonacci series calculated in Chapter 3 to a
# function of the limit 'n'. The triply quoted string here is
# called a docstring and can be read in a standardized program
# as an explanation of the function.

def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a + b
    print()

fib(100)

# The execution of a function introduces a new symbol table
# used for local variables of the function. All variable
# assignments in the function store the value in the local
# symbol table. This is in contrast with variable references
# which first look in the local symbol table, then the local
# symbol table of enclosing functions then in the global
# table and finally in the built-in names.

n = 200
fib(n)
fib(100)
# print(a)

# Functions themselves associate the function name with the
# function object in the current symbol table. The interpreter
# recognizes the object pointed to by that name. Other names
# can point to the same function object.

f = fib
f(200)

# Instead of printing the Fibonacci values, they can be
# returned by the function.

def fib2(n):
    """The Fibonacci function. Returns the series."""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result

fibseries = fib2(400)
print(fibseries)

# The append() function is called a method, and in this case
# it is a method of the list object (result). A method is a
# function that belongs to an object and is called
# object.methodname. Different types define different methods
# so methods of different types can have the same names
# without ambiguity. The user can define object types and
# methods using classes.

# Functions can be defined with variable numbers of
# arguments. For example, by setting default values in
# arguments users are not required to input anything there.

def fib3(n = 10):
    """The Fibonacci series with a default argument value."""
    series = []
    a, b = 0, 1
    while a < n:
        series.append(a)
        a, b = b, a + b
    return series

test = fib3()
print(test)

# Default values are only evaluated once. This makes a
# difference for default arguments that are mutable, such as
# lists, when they are modified by subsequent function calls.

def defaultarg(val, L = []):
    """Test the default argument evaluation."""
    L.append(val)
    return L

print(defaultarg(5))
print(defaultarg(7))
print(defaultarg(9))

L = []
print(defaultarg(1, [1,2,3,2]))

# Default arguments seem similar or equivalent to (?) keyword
# arguments. Keyword arguments take the form kwarg=value, and
# must follow positional arguments. Positional arguments are
# required. Duplicate argument assignments cannot be given,
# and keyword arguments cannot be input by the user when
# kwarg does not exist in the function definition. The order
# does not matter for keyword arguments.

# Flexible numbers of arguments may be passed as formal
# parameters or keyword arguments to functions using * or **,
# respectively. * will take a series of position arguments
# beyond the formal parameter list, while ** will receive a
# dictionary of keyword arguments. Note: The order in which
# the keyword arguments are printed here is guaranteed to
# match the input of them.

def flexargs(a, *listargs, **dictargs):
    print(a)
    print('-' * 40)
    for arg in listargs:
        print(arg)
    print('-' * 40)
    for key in dictargs:
        print(key, ':', dictargs[key])

flexargs(1, 'test', 'this', 2, apple='orange', banana='strawberry')

# The special symbols / and * may be used in function
# definitions to separate positional only, positional or
# keyword, and keyword only arguments.

# def fun(pos1, pos2, / pos_or_kwd, *, kwd1, kwd2):

# Note: These special symbols in a function definition
# restrict the type of parameters allowed. For example,
# arguments preceding a / cannot be keyword arguments.
# arguments preceding a / cannot be keyword arguments.
# Separating arguments with the same name by these special
# symbols allows them to be used in multiple ways without
# ambiguity.

def foo(name, /, **kwargs):
    return name in kwargs
res = foo('beef', **{'beef': 'snack'})
print(res)

# Arbitrary numbers of arguments can be accepted by a
# function if the parameter is *args. This is usually placed
# at the end of the parameter list since it scoops up the
# remaining input. Any formal parameters occurring after
# *args are keyword only and cannot be used positionally.

def concat(*args, sep='/'):
    return sep.join(args)
c1 = concat('a', 'b', 'c')
c2 = concat('a', 'b', 'c', sep=' ')
print(c1)
print(c2)

# The * symbol is also used to unpack argument lists.
limits = [3, 7]
print(list(range(*limits)))

# In the same way, ** is used to unpack dictionaries to
# provide keyword arguments.
items = {'apples': 12, 'pears': 2, 'peaches': 3}
def groceries(**kwargs):
    for kw in kwargs:
        print(kw, ':', kwargs[kw])
groceries(**items)

# Lambda functions lambda a, b: a+b adds a and b together.
def increment(a):
    return lambda x: x + a

f = increment(5)
print(f(0))
print(f(10))

# Style for docstrings. First line quick summary without
# repetition of function name. Separate the rest of the
# docstring, if any, by a line of whitespace.

# Function annotations - unecessary metadata.

# PEP 8 for coding style. UpperCamelCase for classes.
# lowercase_with_underscores for functions and methods.
