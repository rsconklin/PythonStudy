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
