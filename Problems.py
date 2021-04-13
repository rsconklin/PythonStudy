## HR: Maximum Element. https://www.hackerrank.com/challenges/maximum-element/problem. Type: Stacks. Date: 4/13/21.
# Faster solution: O(n).
n = int(input())
stack = [0]

for i in range(n):
    op = list(map(int, input().split()))
    if op[0] == 1:
        stack.append(max(op[1], stack[-1]))
    elif op[0] == 2:
        stack.pop()
    else:
        print(stack[-1])

# My original solution. This is too slow and times out on HR. Others in discussion suggest changing
# the stack order (not according to the queries) with a faster max lookup.
def getMax(operations):
    from heapq import heappush, heappop, heapify
    stack = []
    maxes = []
    maxheap = []
    heapify(maxheap)
    for op in operations:
        if op[0] == '1':
            o = list(map(int, op.split()))
            stack.append(o[1])
            heappush(maxheap, -o[1])
        elif op[0] == '2':
            stack.pop()
        else:
            while -maxheap[0] not in stack:
                heappop(maxheap)
            maxes.append(-maxheap[0])
    return maxes

## HR: Jesse and Cookies. https://www.hackerrank.com/challenges/jesse-and-cookies/problem. Type: Heap. Date: 4/13/21.
def cookies(k, A):
    
    from heapq import heapify, heappop, heappush

    heapify(A)
    count = 0
    heap = A

    while heap[0] < k and count < n - 1:
        c1 = heappop(heap)
        c2 = heappop(heap)
        heappush(heap, c1 + 2*c2)
        count += 1

    if count == n - 1 and heap[0] < k:
        return -1
    else:
        return count

## HR: QHEAP1. https://www.hackerrank.com/challenges/qheap1/problem. Type: Heap. Date: 4/13/21.
# There are n commands, so this is at least O(n). Insertions and removals from sets are O(1), 
# I think, so the most time-consuming command is finding the minimum. When a heap is available,
# the minimum is heap[0] so this is O(1), but I'm not sure what it is if elements are popped
# and the heap needs to be re-heapified. 
from heapq import heappush, heappop

n = int(input())

heap = []
s = set()

for i in range(n):
    q = list(map(int, input().split()))
    if q[0] == 1:
        heappush(heap, q[1])
        s.add(q[1])
    elif q[0] == 2:
        s.remove(q[1])
    else:
        while heap[0] not in s:
            heappop(heap)
        print(heap[0])

## HR: Left Rotation. https://www.hackerrank.com/challenges/array-left-rotation/problem. Type: Arrays. Date: 4/13/21.
def rotateLeft(d, arr):
    return [arr[(i+d)%n] for i in range(n)]

## HR: Dynamic Array. https://www.hackerrank.com/challenges/dynamic-array/problem. Type: Arrays. Date: 4/13/21.
arr = [[] for i in range(n)]
lastAnswer = 0
res = []
for i in range(q):
    idx = (queries[i][1] ^ lastAnswer)%n
    if queries[i][0] == 1:
        arr[idx].append(queries[i][2])
    else:
        ydex = queries[i][2] % len(arr[idx])
        lastAnswer = arr[idx][ydex]
        res.append(lastAnswer)
return res

## HR: 2D Array - DS: https://www.hackerrank.com/challenges/2d-array/problem. Type: Arrays. Date: 4/13/21.
hours = []
for i in range(4):
    for j in range(4):
        top = sum(arr[i][j:j+3])
        mid = arr[i+1][j+1]
        bot = sum(arr[i+2][j:j+3])
        hours.append(top + mid + bot)
return max(hours)

## HR: Arrays - DS: https://www.hackerrank.com/challenges/arrays-ds/problem. Type: Arrays. Date: 4/13/21.
n = int(input())

l = list(map(int, input().split()))
l.reverse()
print(*l)

## HR: Validating and Parsing Email Addresses. https://www.hackerrank.com/challenges/validating-named-email-addresses/problem. Type: RegEx. Date: 4/13/21.
import re

n = int(input())

p = re.compile(r'^<[a-zA-Z](\w|[.]|[-])+@[a-zA-Z]+[.][a-zA-Z]{1,3}>')

for i in range(n):
    name, email = input().split()
    m = p.match(email)
    if m:
        print(name, email)
    else:
        continue

## HR: Group(), Groups(), GroupDict(). https://www.hackerrank.com/challenges/re-group-groups/problem. Type: RegEx. Date: 4/13/21.
import re

s = input()

p = re.compile(r'([a-zA-Z0-9])\1')
m = p.search(s)

if m:
    print(s[m.start()])
else:
    print('-1')

## HR: Re.split(). https://www.hackerrank.com/challenges/re-split/problem. Type: RegEx. Date: 4/13/21.
regex_pattern = r"[.]|[,]"

## HR: Detect Floating Point Number. https://www.hackerrank.com/challenges/introduction-to-regex/problem. Type: RegEx. Date: 4/13/21.
import re

p = re.compile('[+-]?\d*[.]\d+')

n = int(input())

for i in range(n):
    try:
        sequence = input()
        float(sequence)
        m = p.match(sequence)
        if m:
            print('True')
        else:
            print('False')
    except:
        print('False')
        continue

## HR: Incorrect Regex. https://www.hackerrank.com/challenges/incorrect-regex/problem. Type: RegEx. Date: 4/13/21.
import re

n = int(input())

for i in range(n):
    try:
        re.compile(input())
        print('True')
    except:
        print('False')
        continue

## HR: Matrix Script. https://www.hackerrank.com/challenges/matrix-script/problem. Type: RegEx. Date: 4/13/21.
import re

n, m = map(int, input().split())
code = []

for i in range(n):
    code.append(input())

codeline = [code[i][j] for j in range(m) for i in range(n)]
matrix = ''.join(codeline)

print(re.sub(r'\b\W+\b', ' ', matrix))

## HR: Drawing Book. https://www.hackerrank.com/challenges/drawing-book/problem. Type: Algorithms, Implementation. Date: 4/10/21. Time: 00:37:59
# Inspired by HR discussion:
n = int(input())
p = int(input())

if n%2 == 0:
    print(int(min(p/2, (n - p + 1)/2)))
else:
    print(int(min(p/2, (n - p)/2)))

# Original solution. I'm sorry, world. Not enough finesse.
def pageCount(n, p):
    diff = n - p
    if n%2 == 0:
        if p%2 == 0 and p//2 <= diff//2:
            return p//2
        elif p%2 != 0 and p//2 <= (diff + 1)//2:
            return p//2
        else:
            if p%2 == 0:
                return diff//2 
            else:
                return (diff + 1)//2
    else:
        if p == 1:
            return 0
        elif p//2 <= diff//2:
            return p//2
        elif p == n or p == n - 1:
            return 0
        else:
            return diff//2

## HR: Sales by Match. https://www.hackerrank.com/challenges/sock-merchant/problem. Type: Algorithms, Implementation. Date: 4/10/21. Time: 00:01:30.
# O(n).
def sockMerchant(n, ar):
    count = Counter(ar)
    pairs = 0
    for k in count:
        pairs += count[k]//2
    return pairs

## HR: Default Arguments. https://www.hackerrank.com/challenges/default-arguments/problem. Type: Debugging. Date: 4/10/21.
# It is clearer to reset the constructor. Apparently when no argument is given, the default argument class is still called,
# but with the value from the previous call stored.
def print_from_stream(n, stream=EvenStream()):
    stream.__init__()
    for _ in range(n):
        print(stream.get_next())

# My original solution.
def print_from_stream(n, stream=EvenStream()):
    if stream.current%2 == 0:
        stream.current = 0
    for _ in range(n):
        print(stream.get_next())

## HR: Words Score. https://www.hackerrank.com/challenges/words-score/problem. Type: Debugging. Date: 4/10/21.
# The bug to fix was ++score -> score += 1
def is_vowel(letter):
    return letter in ['a', 'e', 'i', 'o', 'u', 'y']

def score_words(words):
    score = 0
    for word in words:
        num_vowels = 0
        for letter in word:
            if is_vowel(letter):
                num_vowels += 1
        if num_vowels % 2 == 0:
            score += 2
        else:
            score += 1
    return score


n = int(input())
words = input().split()
print(score_words(words))

## HR: Reduce Function. https://www.hackerrank.com/challenges/reduce-function/problem. Type: Python Functionals. Date: 4/10/21.
from fractions import Fraction
from functools import reduce

def product(fracs):
    t = Fraction(reduce(lambda x, y: x*y, fracs))
    return t.numerator, t.denominator

## HR: Validating Email Addresses with a Filter. https://www.hackerrank.com/challenges/validate-list-of-email-address-with-filter/problem. Type: Python Functionals. Date: 4/10/21.
# O(n).
def fun(s):
    
    uwlen = len(s.split('@'))
    if uwlen != 2:
        return None
    else:
        username, website_and_ext = s.split('@')
    
    welen = len(website_and_ext.split('.'))
    if welen != 2:
        return None
    else:
        website, extension = website_and_ext.split('.')
    
    usefilt = ''.join([c for c in username if c.isalnum() or c in ('_', '-')])
    webfilt = ''.join([c for c in website if c.isalnum()])
    extfilt = ''.join([c for c in extension if c.isalpha()])
    
    uselen = len(username)
    weblen = len(website)
    extlen = len(extension)
    
    if uselen > 0 and weblen > 0 and 0 < extlen <= 3:
        if usefilt == username and webfilt == website and extfilt == extension:
            return s
        else:
            pass
    else:
        pass

## HR: ginortS. https://www.hackerrank.com/challenges/ginorts/problem. Type: Built-Ins. Date: 4/10/21.
s = input()

ginorts = sorted(s, key=lambda s: (s.isdigit() and int(s)%2 == 0, s.isdigit() and int(s)%2 != 0, s.isupper(), s.islower(), s))

print(*ginorts, sep='')

## HR: Athlete Sort. https://www.hackerrank.com/challenges/python-sort-sort/problem. Type: Built-Ins. Date: 4/10/21.
n, m = map(int, input().split())

athletes = []
for i in range(n):
    athletes.append(list(map(int, input().split())))

k = int(input())

athletes_sorted = sorted(athletes, key=lambda athletes: athletes[k])

for i in range(n):
    print(*athletes_sorted[i])

## HR: Classes: Dealing with Complex Numbers. https://www.hackerrank.com/challenges/class-1-dealing-with-complex-numbers/problem. Type: Classes. Date: 4/10/21.
import math

class Complex(object):
    
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
        
    def __add__(self, no):
        return Complex(self.real + no.real, self.imaginary + no.imaginary)
    
    def __sub__(self, no):
        return Complex(self.real - no.real, self.imaginary - no.imaginary)
    
    def __mul__(self, no):
        return Complex(self.real*no.real - self.imaginary*no.imaginary, self.real*no.imaginary + self.imaginary*no.real)
    
    def __truediv__(self, no):
        try:
            return self.__mul__(Complex(no.real/(no.mod().real)**2, -no.imaginary/(no.mod().real)**2))
        except ZeroDivisionError as e:
            print(e)
            return None

    def mod(self):
        return Complex(math.sqrt(self.real**2 + self.imaginary**2), 0)

## HR: Time Delta. https://www.hackerrank.com/challenges/python-time-delta/problem. Type: Date and Time. Date: 4/10/21.
from datetime import datetime as dt

format = '%a %d %b %Y %H:%M:%S %z'
for i in range(int(input())):
    time1 = dt.strptime(input(), format)
    time2 = dt.strptime(input(), format)
    print(int(abs((time1 - time2).total_seconds())))

## HR: Piling Up!. https://www.hackerrank.com/challenges/piling-up/problem. Type: Collections. Date: 4/10/21.
# O(n).
from collections import deque

for i in range(int(input())):
    
    n = int(input())
    nums = deque(map(int, input().split()))
    
    if nums[0] >= nums[-1]:
        cube = nums.popleft()
    else:
        cube = nums.pop()
    
    while len(nums):
        if nums[-1] <= nums[0] <= cube:
            cube = nums.popleft()
        elif nums[0] < nums[-1] <= cube:
            cube = nums.pop()
        else:
            possible = 0
            break
        possible = 1
    
    if possible:
        print('Yes')
    else:
        print('No')

## HR: Company Logo. https://www.hackerrank.com/challenges/most-commons/problem. Type: collections. Date: 4/09/21.
# Counter is O(n) while sort is O(n log n). Overall O(n log n).
from collections import Counter

count = Counter(input()).items()
for l, n in sorted(count, key = lambda count: (-count[1], count[0]))[:3]:
    print(l, n)

## HR: Word Order. https://www.hackerrank.com/challenges/word-order/problem. Type: collections. Date: 4/09/21.
from collections import Counter

n = int(input())
words = []

for i in range(n):
    words.append(input())

count = Counter(words)
print(len(count))
vals = [count[k] for k in count]
print(*vals, sep = ' ')

## HR: Iterables and Iterators. https://www.hackerrank.com/challenges/iterables-and-iterators/problem. Type: itertools. Date: 4/09/21.
# I think combinations() is O(n), while my probability calculation is O(n). Total = O(n).
from itertools import combinations

n = int(input())
letters = input().split()
k = int(input())

comb = list(combinations(letters, k))
num = [1 for i in comb if 'a' in i]
print(len(num)/len(comb))

## HR: Compress the String!. https://www.hackerrank.com/challenges/compress-the-string/problem. Type: itertools. Date: 4/09/21.
# O(n)
import itertools

n = input()

group = [(len(list(g)), int(k)) for k, g in itertools.groupby(n)]
print(*group)

## HR: Find the Torsional Angle. https://www.hackerrank.com/challenges/class-2-find-the-torsional-angle/problem. Type: Classes. Date: 4/09/21.
import math

class Points(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, no):
        return Points(self.x - no.x, self.y - no.y, self.z - no.z)

    def dot(self, no):
        return self.x*no.x + self.y*no.y + self.z*no.z

    def cross(self, no):
        return Points(self.y*no.z - self.z*no.y, self.z*no.x - self.x*no.z, self.x*no.y - self.y*no.x)

    def absolute(self):
        return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)

## HR: Merge the Tools!. https://www.hackerrank.com/challenges/merge-the-tools/problem. Type: Strings. Date: = 4/09/21.
# Note: I think there is a much more concise (but fundamentally similar) solution with OrderDict from collections.
def merge_the_tools(string, k):

    n = len(string)
    m = int(n/k)
    
    for i in range(1, m + 1):

        udup = list(string[k*(i-1):k*i])
        ui = udup[0]
        uset = set(ui)
        
        for j in range(1, len(udup)):
            if udup[j] in uset:
                pass
            else:
                ui += udup[j]
                uset.add(udup[j])
        print(ui)

## LeetCode (LC): Two Sum II. https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/. Type: Arrays. Date: = 4/09/21
# From LC fastest solution.
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        
        left = 0
        right = len(numbers) - 1
        
        while left != right:
            total = numbers[left] + numbers[right]
            if total == target:
                return [left + 1, right + 1]
            elif total > target:
                right -= 1
            else:
                left += 1

# Faster, because of skipping the inner loop if a duplicate in 'i' is found. But still slow compared to other users.
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        nums = numbers
        numset = set()
        for i in range(len(nums)):
            if nums[i] in numset:
                continue
            else:
                numset.add(nums[i])
            for j in range(-1, -len(nums) + i, -1):
                if nums[i] + nums[j] == target:
                    return [i + 1, len(nums) + j + 1]

# This solution is too slow. It times out on a large array with many repeated entries.
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        for i in range(len(numbers)):
            for j in range(-1, -len(numbers) + i, -1):
                if numbers[i] + numbers[j] == target:
                    return [i + 1, len(numbers) + j + 1]

## LeetCode (LC): Two Sum. https://leetcode.com/problems/two-sum/submissions/. Type: Arrays. DateTime = 4/09/21 1:02.
# O(n)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashTable = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in hashTable:
                return [hashTable[diff], i]
            else:
                hashTable[nums[i]] = i
# O(n^2)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]

## HackerRank (HR): The Minion Game. https://www.hackerrank.com/challenges/the-minion-game/problem. Type: Strings. DateTime = 4/09/21 12:34.
# O(n) version.
def minion_game(string):
    
    kevin, stuart = 0, 0
    slen = len(string)
    
    for i in range(slen):
        if string[i] in {'A', 'E', 'I', 'O', 'U'}:
            kevin += slen - i
        else:
            stuart += slen - i
        
    if kevin == stuart:
        print('Draw')
    elif kevin > stuart:
        print('Kevin', kevin)
    elif kevin < stuart:
        print('Stuart', stuart)

# Brute force version. O(n^2). Times out on HR. 
def minion_game(string):
    
    kevin, stuart = 0, 0
    slen = len(string)
    
    for substrlen in range(1, slen + 1):
        for start in range(slen + 1 - substrlen):
            substr = string[start:start + substrlen]
            # print(substrlen, start, substr) This tests the indexing.
            if substr[0] in {'A', 'E', 'I', 'O', 'U'}:
                kevin += 1
            else:
                stuart += 1
            
    if kevin == stuart:
        print('Draw')
    elif kevin > stuart:
        print('Kevin', kevin)
    elif kevin < stuart:
        print('Stuart', stuart)
