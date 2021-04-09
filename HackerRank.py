## HR: Merge the Tools!. https://www.hackerrank.com/challenges/merge-the-tools/problem. Type: Strings. Date = 4/09/21.

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
