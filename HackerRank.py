## LeetCode (LC): Two Sum. https://leetcode.com/problems/two-sum/submissions/. Type: Arrays. DateTime = 4/09/21 1:02.

# Brute force. Yet to optimize. Attempted list filtration at the start.
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numsfilt = nums #[x for x in nums if x <= target]
        for i in range(len(numsfilt) - 1):
            for j in range(i + 1, len(numsfilt)):
                if numsfilt[i] + numsfilt[j] == target:
                    print(numsfilt[i], numsfilt[j])
                    i1 = nums.index(numsfilt[i])
                    i2 = nums.index(numsfilt[j], i1 +1, len(nums))
                    return [i1, i2]
                else:
                    continue

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
