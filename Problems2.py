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
