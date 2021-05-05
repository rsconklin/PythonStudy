## LC: Reorder Data in Log Files. https://leetcode.com/problems/reorder-data-in-log-files/. Type: Strings, Sort. Date: 5/05/21.
# O(k * nlogn)
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        
        letterlogs = []
        digitlogs = []
        
        for i in logs:
            if i.split()[1].isalpha():
                letterlogs.append(i)
            else:
                digitlogs.append(i)
        
        letterlogs.sort(key = lambda x: (x.split()[1:], x.split()[0]))
        
        return letterlogs + digitlogs

## LC: LRU Cache. https://leetcode.com/problems/lru-cache/. Type: Caches. Date: 5/05/21.
# O(capacity) for put().
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.keytimes = {}
        self.time = 0

    def get(self, key: int) -> int:
        if key in self.cache:
            self.keytimes[key] = self.time
            self.time += 1
            return self.cache[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.keytimes[key] = self.time
            self.time += 1
        else:
            if len(self.cache) == self.capacity:
                time = self.time + 1
                for k, t in self.keytimes.items():
                    if time > t:
                        delkey = k
                        time = t
                self.cache.pop(delkey)
                self.keytimes.pop(delkey)
                self.cache[key] = value
                self.keytimes[key] = self.time
                self.time += 1
            else:
                self.cache[key] = value
                self.keytimes[key] = self.time
                self.time += 1

## LC: Diameter of Binary Tree. https://leetcode.com/problems/diameter-of-binary-tree/. Type: Trees. Date: 5/05/21.
# O(n)
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        
        self.diameter = 0
        
        self.dFind(root)
        return self.diameter
    
    def dFind(self, node):
        if not node:
            return 0
        left = self.dFind(node.left)
        right = self.dFind(node.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)

## LC: Reorganize String. https://leetcode.com/problems/reorganize-string/. Type: Strings. Date: 5/05/21.
# O(nlogn)
from collections import Counter

class Solution:
    def reorganizeString(self, S: str) -> str:
        count = Counter(S)
        s = list(S)
        l = len(S)
        
        s.sort(key = lambda x: (count[x], x), reverse = True)
        
        if count[s[0]] > (l + 1) // 2:
            return ''
        
        iters = iter(s)
        result = [None] * l
        
        for i in range(0, l, 2):
            result[i] = next(iters)
        for i in range(1, l, 2):
            result[i] = next(iters)
            
        return ''.join(result)

## LC: Longest Increasing Subsequence. https://leetcode.com/problems/longest-increasing-subsequence/. Type: Arrays. Date: 5/05/21.
# O(nlogn)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        pSorted = [[nums[0]]]
        
        for num in nums[1:]:
            for i in range(len(pSorted)):
                if num <= pSorted[i][-1]:
                    pSorted[i].append(num)
                    break
                elif i == len(pSorted) - 1:
                    pSorted.append([num])
            
        return len(pSorted)

## LC: Find Median from Data Stream. https://leetcode.com/problems/find-median-from-data-stream/. Type: Heaps, Classes. Date: 5/04/21.
# O(nlogn)
from heapq import *

class MedianFinder:

    def __init__(self):
        self.lower = []
        self.upper = []

    def addNum(self, num: int) -> None:
        if len(self.lower) < len(self.upper):
            heappush(self.lower, -heappushpop(self.upper, num))
        else:
            heappush(self.upper, -heappushpop(self.lower, -num))

    def findMedian(self) -> float:
        if len(self.lower) < len(self.upper):
            return self.upper[0]
        else:
            return (self.upper[0] - self.lower[0])/2

## LC: Word Search II. https://leetcode.com/problems/word-search-ii/. Type: Graph, Dynamic Programming (DP). Date: 5/04/21.
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        m = len(board)
        n = len(board[0])
        wordlist = set()
        
        for w in words:
            for i in range(m):
                for j in range(n):
                    if board[i][j] == w[0]:
                        if self.wordtest(board, m, n, w, i, j):
                            wordlist.add(w)
        
        return list(wordlist)
        
    def wordtest(self, board, m, n, w, i, j):
        stack = [(0, i, j)]
        indexset = {(i, j)}
        visited = set()
        
        while stack:
            lindex, i, j = stack[-1]
            visited.add(stack[-1])
            
            if lindex == len(w) - 1:
                return True
            
            if i - 1 >= 0 and ((lindex + 1, i - 1, j) not in visited) and ((i - 1, j) not in indexset):
                if board[i - 1][j] == w[lindex + 1]:
                    stack.append((lindex + 1, i - 1, j))
                    indexset.add((i - 1, j))
                    continue
            if j + 1 < n and ((lindex + 1, i, j + 1) not in visited) and ((i, j + 1) not in indexset):
                if board[i][j + 1] == w[lindex + 1]:
                    stack.append((lindex + 1, i, j + 1))
                    indexset.add((i, j + 1))
                    continue
            if i + 1 < m and ((lindex + 1, i + 1, j) not in visited) and ((i + 1, j) not in indexset):
                if board[i + 1][j] == w[lindex + 1]:
                    stack.append((lindex + 1, i + 1, j))
                    indexset.add((i + 1, j))
                    continue
            if j - 1 >= 0 and ((lindex + 1, i, j - 1) not in visited) and ((i, j - 1) not in indexset):
                if board[i][j - 1] == w[lindex + 1]:
                    stack.append((lindex + 1, i, j - 1))
                    indexset.add((i, j - 1))
                    continue
            
            stack.pop()
            indexset.remove((i, j))
        
        return False
                
                        

## LC: Design Add and Search Words Data Structure. https://leetcode.com/problems/design-add-and-search-words-data-structure/. Type: Dictionaries. Date: 5/04/21.
# RegEx Slow.
class WordDictionary:

    import re
    
    def __init__(self):
        self.dict = {}
        

    def addWord(self, word: str) -> None:
        try:
            self.dict[len(word)].append(word)
        except:
            self.dict[len(word)] = [word]

    def search(self, word: str) -> bool:    
        p = re.compile(word)
        if len(word) in self.dict:
            for i in self.dict[len(word)]:
                m = p.match(i)
                if m:
                    return True
        return False

## LC: Course Schedule II. https://leetcode.com/problems/course-schedule-ii/. Type: Graphs. Date: 5/04/21.
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        '''Return the topological sort of the courses with their prerequisites.
        
        Our strategy will be to build an adjacency list of prerequisites, count
        the number of prerequisites for each course, isolate those with zero
        prerequisites, pop these from the list while marking their order, and
        continue until all courses have been ordered or there is a loop. If a
        loop is encountered, return an empty array.'''
        
        from collections import deque
        
        # Adjacency list. O(p)
        adjList = [[] for i in range(numCourses)]
        for p in prerequisites:
            adjList[p[1]].append(p[0])
        
        # Number of prerequisites for each course. O(p)
        numPre = [0 for i in range(numCourses)]
        for p in prerequisites:
            numPre[p[0]] += 1
        
        # Courses without prerequisites.
        deq = deque([])
        for i in range(numCourses):
            if numPre[i] == 0:
                deq.append(i)
                
        # The container for the final topologically sorted courses.
        topSort = []
        
        # Pop courses with no prerequisites and reduce the numPre count. O(p)
        while deq:
            c = deq.popleft()
            topSort.append(c)
            for course in adjList[c]:
                numPre[course] -= 1
                if numPre[course] == 0:
                    deq.append(course)
        
        if len(topSort) == numCourses:
            return topSort
        else:
            return []

## LC: Number of Islands. https://leetcode.com/problems/number-of-islands/. Type: Arrays. Date: 5/03/21.
# O(n*m)
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        m = len(grid)
        n = len(grid[0])
        islands = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.submerge(grid, (i, j))
                    islands += 1
        
        return islands
            
    # Terraform in a clockwise direction. 
    # If a dead end is hit, pop the stack to reverse.
    def submerge(self, grid, coords):
        
        i, j = coords
        grid[i][j] = '0'
        stack = [(i, j)]
        m = len(grid)
        n = len(grid[0])
        
        while stack:
            i, j = stack[-1]
            if i - 1 >= 0:
                if grid[i - 1][j] == '1':
                    stack.append((i - 1, j))
                    i, j = i - 1, j
                    grid[i][j] = '0'
                    continue
            if j + 1 < n:
                if grid[i][j + 1] == '1':
                    stack.append((i, j + 1))
                    i, j = i, j + 1
                    grid[i][j] = '0'
                    continue
            if i + 1 < m:
                if grid[i + 1][j] == '1':
                    stack.append((i + 1, j))
                    i, j = i + 1, j
                    grid[i][j] = '0'
                    continue
            if j - 1 >= 0:
                if grid[i][j - 1] == '1':
                    stack.append((i, j - 1))
                    i, j = i, j - 1
                    grid[i][j] = '0'
                    continue
            stack.pop()

## LC: Min Stack. https://leetcode.com/problems/min-stack/. Type: Classes. Date: 5/03/21.
# Very slow. Need to optimize the getMin() method.
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, val: int) -> None:
        self.stack = self.stack + [val]

    def pop(self) -> None:
        self.stack = self.stack[:-1]

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return min(self.stack)

## LC: Best TIme to Buy and Sell Stock. https://leetcode.com/problems/best-time-to-buy-and-sell-stock/. Type: Arrays. Date: 5/03/21.
# O(n).
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        if not prices:
            return 0
        
        maxProfit = 0
        minBuy = prices[0]
        
        for i in range(1, len(prices)):
            maxProfit = max(maxProfit, prices[i] - minBuy)
            minBuy = min(minBuy, prices[i])
        
        return maxProfit

## LC: Binary Tree Level Order Traversal. https://leetcode.com/problems/binary-tree-level-order-traversal/. Type: Trees. Date: 5/03/21.
# O(n)
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        
        from collections import deque
        
        if not root:
            return None
        
        deq = deque([(root, 0)])
        res = []
        
        while deq:
            current, height = deq.popleft()
            res.append((current.val, height))
            if current.left:
                deq.append((current.left, height - 1))
            if current.right:
                deq.append((current.right, height - 1))
        
        result = [[] for i in range(-height + 1)]
        
        for r in res:
            result[-r[1]].append(r[0])
        
        return result

## LC: Symmetric Tree. https://leetcode.com/problems/symmetric-tree/. Type: Trees. Date: 5/03/21.
# O(n)
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        
        from collections import deque
        
        if not root:
            return True
        
        t1 = root.left
        t2 = root.right
            
        if not t1 and not t2:
            return True
        elif t1 and not t2:
            return False
        elif t2 and not t1:
            return False
        elif t1.val != t2.val:
                return False
        
        node = t1
        deq = deque([(node, 0, 0)])
        res = set()
        while deq:
            current, height, distance = deq.popleft()
            res.add((current.val, height, distance))
            if current.left:
                deq.append((current.left, height - 1, distance - 1))
            if current.right:
                deq.append((current.right, height - 1, distance + 1))
                
        node = t2
        deq = deque([(node, 0, 0)])
        res2 = set()
        while deq:
            current, height, distance = deq.popleft()
            res2.add((current.val, height, distance))
            if current.left:
                deq.append((current.left, height - 1, distance + 1))
            if current.right:
                deq.append((current.right, height - 1, distance - 1))

        print(res)
        print(res2)
        return res == res2

## LC: Same Tree. https://leetcode.com/problems/same-tree/. Type: Trees. Date: 5/03/21.
# O(n)
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        
        # Let's do a breadth-first search keeping track of the tree structure.
        # Do for each root, then compare the final results.
        
        from collections import deque
        
        node = p
        deq = deque([(node, 0, 0)])
        res = set()
        
        if deq[0][0]:
            while deq:
                current, height, distance = deq.popleft()
                res.add((current.val, height, distance))
                if current.left:
                    deq.append((current.left, height - 1, distance - 1))
                if current.right:
                    deq.append((current.right, height - 1, distance + 1))
                    
        node = q
        deq = deque([(node, 0, 0)])
        res2 = set()
        
        if deq[0][0]:
            while deq:
                current, height, distance = deq.popleft()
                res2.add((current.val, height, distance))
                if current.left:
                    deq.append((current.left, height - 1, distance - 1))
                if current.right:
                    deq.append((current.right, height - 1, distance + 1))
            
        return res == res2
