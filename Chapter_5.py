# Data structures. This includes lists, sets, dictionaries,
# tuples. Lists as stacks and queues. List comprehensions,
# nested list comprehensions. Looping techniques.

# There are several built-in list methods.
nums = [1, 2, 3, 4, 5]
print(nums.pop())
print(nums.pop(2))
print(nums)

nums.clear()
print(nums)

nums = [5, 6, 7, 8, 9]
print(nums.index(7, 1, 5))

nums.extend([9, 9, 5, 6])
print(nums)
print(nums.count(9))

# There is a built-in sort() method. This has customizable
# arguments, but also a default function.
nums.sort()
print(nums)

nums.reverse()
print(nums)

# Mixed-type lists cannot be sorted in some cases (when?)
# such as when integers and strings are present. Complex
# numbers cannot be sorted in fullform.

# Methods that only modify the list return None.
print(nums.sort())

# Lists can be used as stacks, where the last element in is
# the first element out.
stack = [1, 2, 3]
stack.append(5)
stack.append(7)
print(stack)
stack.pop()
print(stack)

# Lists can also be used as queues, where the first element
# added is the first element out. To do so, the module
# collections.deque should be used, since this has fast
# appends and pops on both sides.
from collections import deque
queue = deque([4, 5, 6])
queue.append(7)
print(queue.popleft())
print(queue)
queue.appendleft(16)
print(queue)
listfromqueue = list(queue)
print(listfromqueue)

# It seems that list comprehensions can be used in place of
# lambda expressions for creating lists.
squares = [x**2 for x in range(11)]
print(squares)

# Make a half-parabola with list comprehensions.
for i in range(6):
    print('*' * squares[i])

# List comprehensions consist of an expression followed by a
# for clause and then 0 or more for or if clauses.
unequal = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
print(unequal)

# List comprehensions are useful for applying functions on
# each element of lists, and otherwise modifying them and
# filtering them. For example, take the absolute value of
# each element in a list, flatten the list, remove negative
# values from a list.
nums = [1, 2, -6, 7, -4, -8, 10]
numspos = [num for num in nums if num > 0]
print(numspos)
numsabs = [abs(num) for num in nums]
print(numsabs)
numsnest = [[1, 2, 3], [4, 5, 6], [7]]
numsflat = [num for ele in numsnest for num in ele]
print(numsflat)

# Square each element in a list and make a list of tuples
# with the original and the squared values.
nums = [1, 2, 3, 4, 5]
numssquared = [(num, num**2) for num in nums]
print(numssquared)

# Delete an element from a list according to its index rather
# than its value.
del nums[2]
print(nums)

# Tuples are comma separated values, and are immutable.
t = 1, 2, 3
print(t[1])

# Tuples may be nested.
u = t, 5
print(u)

# Tuples may contain immutable objects.
t2 = [1, 2, 3], [4, 5, 6]
print(t2)
t2[0][2] = 5
print(t2)

# Tuples can be packed and unpacked. Creating a tuple from
# a sequence of comma separated values is packing.
# Extracting the elements of a tuple by doing the inverse
# operation is unpacking.
t = 'unpack', 'this', 'tuple'
print(t)
x, y, z = t
print(x, y, z)

# The method of unpacking works for any sequence. Multiple
# assignment is really just tuple packing and unpacking.

# A set is an unordered collection of unique elements. Basic
# uses include membership testing and eliminating
# duplicates. An empty set is created with set() rather than
# {} since the latter creates an empty dictionary.
recipe = {'bread crumbs', 'salt', 'pepper', 'butter'}
print('salt' in recipe)

# Sets have corresponding mathematical operations according
# to set theory. For example union, intersection,
# difference, etc.
a = {5, 1, 1, 'two', 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
print(a)
print('a not b:', a - b)
print('a and b:', a & b)
print('a or b or both:', a | b)
print('a or b not both:', a ^ b)

# Set comprehensions are also supported. The following gives
# 'a and b'.
c = {x for x in b if x in a}
print(c)

# Dictionaries are indexed by keys rather than numbers. The
# keys can be any immutable types, such as strings and
# numbers, or tuples that are purely immutable. A dictionary
# is a set of key: value pairs, with the requirement that
# the keys be unique.

d = {1: 'Hello', 2: 'Fresh'}
print(list(d))
print(1 in d, 3 in d)
d[7] = 'Lettuce'
print(d)
d = {1: [1, 2, 3], 2: [4, 5, 6]}
d[3] = [7, 8, 9]
d[1].append(4)
print(d)

# There are also dictionary comprehensions.
d = {x: x**2 for x in [1, 2, 3]}
print(d)

for k, v in d.items():
    print(k, v)

for index, value in enumerate(['Fresh', 'Prince', 'Hams']):
    print(index, value)

nums1 = [1, 2, 3]
nums2 = [4, 5, 6]
for i, j in zip(nums1, nums2):
    print(i, j)

# 'NaN' means 'Not a number'.

# For conditions in loops, the condition operators, such as
# in, not in, is, and is not, have lower priority than all
# numerical operators. Comparisons may be chained.
print(1 < 2 == 2)

# For comparison operators, not has the highest priority
# while or has the lowest. So A and not B or C is equal to
# (A and (not B)) or C.

# Assignment within expressions must be done explicitly with
# the walrus operator.

# Sequences can be compared when they are of the same type.
print((1, 2, 3) < (1, 2, 4))
print('abc' < 'c' < 'pascal' < 'python')
print((1, 2) < (1, 2, -1))

