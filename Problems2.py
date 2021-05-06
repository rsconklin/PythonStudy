## LC: Minimum Difficulty of a Job Schedule. https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/. Type: Dynamic Programming. Date: 5/06/21.
# O(n^2 * d)
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        
        numjobs = len(jobDifficulty)
        
        # If there are not enough jobs to fill the days.
        if numjobs < d:
            return -1
        
        inf = float('inf')
        
        # The final dp[daysleft][i] will equal the minimum job difficulty
        # for starting with daysleft = the number of days remaining to
        # complete the jobs and i = the index of the first job to be
        # started.
        dp = [[inf] * numjobs + [0] for daysleft in range(d + 1)]
        
        for daysleft in range(1, d + 1):
            # The range here is from the first job to the last possible
            # job given that one job must be left for each remaining day
            # of work. This is defining daysleft = 1 to mean that there
            # is one slot available in which all the jobs must be
            # completed.
            for i in range(numjobs - daysleft + 1):
                daymax = 0
                for j in range(i, numjobs - daysleft + 1):
                    daymax = max(daymax, jobDifficulty[j])
                    dp[daysleft][i] = min(dp[daysleft][i], dp[daysleft - 1][j + 1] + daymax)
                    
        return dp[d][0]

## LC: Maximum Area of a Piece of Cake after Horizontal and Vertical Cuts.
## https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts/. Type: Arrays. Date: 5/06/21.
# O(nlogn) - I sort.
class Solution:
    def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
        horizontalCuts.sort()
        verticalCuts.sort()
        
        hGap = max(horizontalCuts[0], h - horizontalCuts[-1])
        for i in range(1, len(horizontalCuts)):
            dif = horizontalCuts[i] - horizontalCuts[i - 1]
            if dif > hGap:
                hGap = dif
        
        vGap = max(verticalCuts[0], w - verticalCuts[-1])
        for i in range(1, len(verticalCuts)):
            dif = verticalCuts[i] - verticalCuts[i - 1]
            if dif > vGap:
                vGap = dif
        
        return (hGap * vGap) % (10**9 + 7)

## LC: Robot Bounded in Circle. https://leetcode.com/problems/robot-bounded-in-circle/. Type: ?. Date: 5/06/21.
# O(n)
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        '''The robot's trajectory is bounded if...
        
        1. The instructions lead it back to the start after one sequence.
        2. The final direction of the robot is different from the initial
            direction. (The robot's paths can be thought of vectors that
            rotate and cancel each other after four cycles.)'''
        
        # Keep track of coordinates to see whether the robot arrives back
        # at the origin. The directions correspond to how the coordinates
        # should be incremented if the robot is facing that direction and
        # moves forward. For example, the y-coordinate goes up by one and
        # the x-coordinate does not change when the robot moves forward
        # while facing north, hence (0, 1).
        pos = [0, 0]
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dindex = 0
        
        # Iterate through the commands.
        for i in instructions:
            if i == 'G':
                pos[0] += direction[dindex][0]
                pos[1] += direction[dindex][1]
            elif i == 'R':
                dindex = (dindex + 1) % 4
            elif i == 'L':
                dindex = (dindex + 3) % 4
        
        if pos == [0, 0] or dindex != 0:
            return True
        
        return False

## LC: LRU Cahe. https://leetcode.com/problems/lru-cache/. Type: Data Structures. Date: 5/05/21.
# O(1) for put() and get(). Doubly-linked list.
class Node:
    '''For a doubly-linked list with dictionary attributes.'''
    
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.val
        else:
            return -1
            
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node
        self._add(node)
        if len(self.cache) > self.capacity:
            node = self.head.next
            self._remove(node)
            del self.cache[node.key]
    
    def _remove(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p
        
    def _add(self, node):
        p = self.tail.prev
        p.next = node
        self.tail.prev = node        
        node.prev = p
        node.next = self.tail

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
