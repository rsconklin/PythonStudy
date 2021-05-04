## LC: Number of Islands. https://leetcode.com/problems/number-of-islands/. Type: Arrays. Date: 5/03/21.
# O(n^2)
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
