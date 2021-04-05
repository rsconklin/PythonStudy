# Hash characters within strings are just hash characters.

# Division always returns a floating point number.

# To do floor division, where any fractional result is discarded, use the // operator.
# To calculate the remainder, take the mod %.

print(7 // 3)
print(7 % 3)
print((7 // 3 + 7 % 3) == int(7/3))
print((7 // 3 + 7 % 3))
print(int(7/3))
print(7/3)
print(7 == (7//3)*3 + 7 % 3)

# Powers input using the ** operator.

print(5 ** 3)

# Full support for floating point. Given mixed integer and floating point input, the
# integer part gets converted to float.

print(5 + 1.7)

# In interactive mode, the last printed expression is assigned to the variable '_'.
# This is similar to the % command in Mathematica.

# Round a floating point number.

print(round(7.123456, 2))

# Python has built-in support for complex numbers. Denote the imaginary part by j
# or J.

print(abs(1 + 1j))
print(abs(1 + 1J))

# Strings enclosed equivalently in '' or "". Use \ to escape quotes.

print("Testing contractions such as don\'t.")
print('Mixing "quotation" types.')

# The print function removes enclosing quotation marks and prints escaped and special
# characters.

print('Two \nlines.')

# To cause characters prefaced by \ to not be interpreted as special characters, use
# raw strings by adding an 'r' before the first quote.

print(r'C:\some\name')

# End of lines are automatically included in strings, but this can be prevented with
# \. Can use triple quotes to write string literals over multiple lines.

print('''\
I would like to
    write this message
    over many lines.\
''')

# Concatenate strings with + or repeat them with *.

print("Repeat after me: " + 'Hooray! '*3)

# String literals (ones enclosed in quotes) next to each other are automatically
# concatenated.

print('Add' ' all' ' these ' 'strings.')

# This feature of concatenating adjacent strings is very useful for breaking long
# strings, which can be done by grouping them together in parentheses. Note that this
# only works with string literals, so it is not possible to assign a string to a
# variable and then use this feature. Use + for this.

print(('This is a very long string '
       'that I would like to break '
       'over several lines for readability.'))

# Strings are indexed in the same way as lists. The index 0 is reserved for the first
# element of the list, and -0 = 0, so negative indices start at -1.

howlong = 'How long is this string?'
print(howlong, len(howlong))
print('The string is {} units long.'.format(len(howlong)))
print('What is the fourth letter from the end? It is: "{}".'.format(howlong[-4]))

# Strings can be sliced like lists, with the same syntax that the first element is
# always included and the last is always excluded such that s[:i] + s[i:] is equal to
# s. The indices used in slicing have useful defaults. An empty first index defaults
# to 0, while an empty second to the length of the list.

s = 'Hello, world!'
print(s[:5] + s[5:])
print(s[:-5] + s[-5:])

# The length of a slice is the difference between the indices (simplest case) since
# the first index is counted and the last is excluded.

print(len(s[1:7]) == 6)

# Attempting to access an element not in a list gives an error, but slices are more
# forgiving.

# print(s[100]) gives an error.
print(s[:100])

# Python strings are immutable, so attempting to change then results in an error.
# If a string needs to be changed, a new one should be created.

# Lists also support operations such as concatenation.

values = [1, 2, 3, 4] + [5, 6]
print(values)
print(values + [7, 8])
print(values * 2)

# Lists are mutable so their elements can be changed.

values[2] = 7
print(values)

# Append to a list.

values.append('Hello')
print(values)

# Slices can be used for changing lists.

values[:3] = [10, 9, 8]
print(values)
# Remove elements from a list.
values[-4:] = []
print(values)
# Clear a list.
values[:] = []
print(values)

# Create a multidimensional (nested) list by inserting a list as an element into
# another list. Use a separate index bracket for each dimension to access elements.

newvalues = [1, 2, 3, [4, 5, 6]]
print(newvalues[3][2])

# Note: The exponent operator ** has higher precedence than arithmetic operators such
# that

print(-3**2 == -9)
print((-3)**2 == 9)

# As a first example of a more complicated task, we now write the first several
# numbers in the Fibonacci series. This example has several new features:
# 1) Multiple assignment: More than one variable defined in one command.
# 1.1) The right hand side of assignments is handled first, from left to right.
# 2) Comparison operators are standard. >, <, <=, >=, ==, !=.
# 3) Commands are grouped by indentation, which must be equal for each statement.
# 4) The print function defaults to separating printed items by a space. This can
# be changed to another value by using  end=',', for example.

a, b = 0, 1
while a < 10:
    print(a)
    a, b = b, a + b

a, b = 0, 1
while a < 10:
    print(a, end=', ')
    a, b = b, a + b

print('\nThe value of "a" is', a)
