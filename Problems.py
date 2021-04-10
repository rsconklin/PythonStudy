## HR: Classes: Dealing with Complex Numbers. https://www.hackerrank.com/challenges/class-1-dealing-with-complex-numbers/problem. Type: Classes. 4/10/21.
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

## HR: Time Delta. https://www.hackerrank.com/challenges/python-time-delta/problem. Type: Date and Time. 4/10/21.
from datetime import datetime as dt

format = '%a %d %b %Y %H:%M:%S %z'
for i in range(int(input())):
    time1 = dt.strptime(input(), format)
    time2 = dt.strptime(input(), format)
    print(int(abs((time1 - time2).total_seconds())))

## HR: Piling Up!. https://www.hackerrank.com/challenges/piling-up/problem. Type: Collections. 4/10/21.
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

## HR: Company Logo. https://www.hackerrank.com/challenges/most-commons/problem. Type: collections. Date 4/09/21.
# Counter is O(n) while sort is O(n log n). Overall O(n log n).
from collections import Counter

count = Counter(input()).items()
for l, n in sorted(count, key = lambda count: (-count[1], count[0]))[:3]:
    print(l, n)

## HR: Word Order. https://www.hackerrank.com/challenges/word-order/problem. Type: collections. Date 4/09/21.
from collections import Counter

n = int(input())
words = []

for i in range(n):
    words.append(input())

count = Counter(words)
print(len(count))
vals = [count[k] for k in count]
print(*vals, sep = ' ')

## HR: Iterables and Iterators. https://www.hackerrank.com/challenges/iterables-and-iterators/problem. Type: itertools. Date 4/09/21.
# I think combinations() is O(n), while my probability calculation is O(n). Total = O(n).
from itertools import combinations

n = int(input())
letters = input().split()
k = int(input())

comb = list(combinations(letters, k))
num = [1 for i in comb if 'a' in i]
print(len(num)/len(comb))

## HR: Compress the String!. https://www.hackerrank.com/challenges/compress-the-string/problem. Type: itertools. Date 4/09/21.
# O(n)
import itertools

n = input()

group = [(len(list(g)), int(k)) for k, g in itertools.groupby(n)]
print(*group)

## HR: Find the Torsional Angle. https://www.hackerrank.com/challenges/class-2-find-the-torsional-angle/problem. Type: Classes. Date 4/09/21.
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

## HR: Merge the Tools!. https://www.hackerrank.com/challenges/merge-the-tools/problem. Type: Strings. Date = 4/09/21.
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

## LeetCode (LC): Two Sum II. https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/. Type: Arrays. Date = 4/09/21
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
