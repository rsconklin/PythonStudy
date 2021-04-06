# Formatted string literals a strings prefixed by f or F.
meat = 'chicken'
recipe = 'salad'
print(f'Let us make a {meat} {recipe}.')
print('Let us make a {} {}.'.format(meat, recipe))

# Any value can be converted to a string using repr() or
# str().

import math

print(f'The value of Pi is about {math.pi:.5f}.')

# Passing an integer after the ':' causes the field to be
# that integer number of characters wide.
print(f'{3:70} {7:10}')

# The str.format() method can be used with indices.
print('The first field is {0}, then {1}.'.format('Chicken', 'Egg'))
print('Or was it {1} then {0}?'.format('Chicken', 'Egg'))

d = {1: 'Sandwich', 2: 'Cheese', 3: 'Lettuce'}
print('Eat a {0[2]} and {0[3]} {0[1]}!'.format(d))

# Output a table of formatted integers and their squares and
# cubes.
for x in range(1, 11):
    print('{0:2} {1:3} {2:4}'.format(x, x**2, x**3))

# The % (modulo) operator can also be used for string
# formatting. Each instance of % in the string is replaced
# by zero or more elements of the values following the
# external %.
print('math' in dir())
print('A value of Pi is %.3f.' % math.pi)

# Strings are simply written to and read from files. Numbers
# and other objects are more complicated since the read()
# method only returns strings, which will have to be passed
# to a function like int(). More complex data types, such as
# dictionaries and nested lists, make conversion by hand
# more challenging. For this, there exists the popular data
# interchange format called JSON (JavaScript Object
# Notation) can be used from the standard module json. This
# can take Python data hierarchies and convert them to string
# representations (called serializing). Reconstructing the
# data from the string representation after is called
# deserializing.

import json
jtest = json.dumps([1, 'simple', 'list'])
print(jtest)
# json.dump(x, f) to serialize x to a text file f.
# x = json.load(f) to deserialize for reading.
# Serializing arbitrary class instances in JSON still
# requires a bit of extra effort.
