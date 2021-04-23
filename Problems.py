## LC: Merge k Sorted Lists. https://leetcode.com/problems/merge-k-sorted-lists/. Type: Linked Lists. Date: 4/22-23/21.
# O(max(klogk, m)) where k is the number of linked lists and m is the number of nodes.
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        # Work with a heap to effectively sort the linked lists with respect to each other.
        from heapq import heapify, heappop, heappush
        
        # Deal with the edge cases of empty lists.
        if len(lists) == 0:
            return None
        
        # Construct the heap according to node.val, but keep indices to come back to each ll.
        heap = []
        for i in range(len(lists)):
            if lists[i]:
                heap.append((lists[i].val, i))
        heapify(heap)
        
        if not heap:
            return None
                
        # Get the minimal node from the heap by popping, then replace with node.next and
        # iterate until all lists elements are extracted.
        idx = heappop(heap)[1]
        headnode = lists[idx]
        lists[idx] = lists[idx].next
        if lists[idx]:
            heappush(heap, (headnode.next.val, idx))
        
        node = headnode
        while heap:
            idx = heappop(heap)[1]
            node.next = lists[idx]
            node = lists[idx]
            lists[idx] = lists[idx].next
            if lists[idx]:
                heappush(heap, (lists[idx].val, idx))
            
        return headnode

# O(k^2*m^2) (baaaad) where k is the length of lists and m is the length of each sublist.
# Times out.
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        if lists in ([], [[]]):
            return None
        
        values = []
        
        i = 0
        k = len(lists)
        while i < k:
            if not lists[i]:
                del lists[i]
                k = len(lists)
            else:
                values.append((lists[i].val, i))
                i += 1
        
        if values == []:
            return None
            
        minval, minpos = min(values)
        
        firstNode = ListNode(minval)
        
        lists[minpos] = lists[minpos].next
        
        oldNode = firstNode
        while True:
            values = []
            i = 0
            k = len(lists)
            while i < k:
                if not lists[i]:
                    del lists[i]
                    k = len(lists)
                else:
                    values.append((lists[i].val, i))
                    i += 1
            if values == [] or k == 0:
                break
            minval, minpos = min(values)
            newNode = ListNode(minval)
            oldNode.next = newNode
            oldNode = newNode
            lists[minpos] = lists[minpos].next
            
        return firstNode

## LC: Valid Parentheses. https://leetcode.com/problems/valid-parentheses/. Type: Strings and Dictionary. Date: 4/22/21.
# O(n).
class Solution:
    def isValid(self, s: str) -> bool:
        
        d = {'(' : ')', '[' : ']', '{' : '}'}
        stack = []
        if s == '':
            return True
        elif len(s) == 1:
            return False
        
        for i in s:
            if i in d.keys():
                stack.append(i)
            elif (i in d.values() and stack == []) or (d[stack.pop()] != i):
                return False
            
        if stack == []:
            return True
        else:
            return False

## LC: Roman to Integer. https://leetcode.com/problems/roman-to-integer/. Type: Strings. Date: 4/22/21.
class Solution:
    def romanToInt(self, s: str) -> int:
        
        count = 0
        i = 0
        
        while i < len(s):
            
            if s[i] == 'M':
                count += 1000
                i += 1
            elif s[i : i + 2] == 'CM':
                count += 900
                i += 2
            elif s[i] == 'D':
                count += 500
                i += 1
            elif s[i : i + 2] == 'CD':
                count += 400
                i += 2
            elif s[i] == 'C':
                count += 100
                i += 1
            elif s[i : i + 2] == 'XC':
                count += 90
                i += 2
            elif s[i : i + 2] == 'XL':
                count += 40
                i += 2
            elif s[i] == 'L':
                count += 50
                i += 1
            elif s[i] == 'X':
                count += 10
                i += 1
            elif s[i : i + 2] == 'IX':
                count += 9
                i += 2
            elif s[i : i + 2] == 'IV':
                count += 4
                i += 2
            elif s[i] == 'V':
                count += 5
                i += 1
            elif s[i] == 'I':
                count += 1
                i += 1
                
        return count

## LC: Integer to Roman. https://leetcode.com/problems/integer-to-roman/. Type: Strings. Date: 4/22/21.
# O(n)
class Solution:
    def intToRoman(self, num: int) -> str:
        
        numerals = []
        
        # Start with the largest numbers and subtract these from the total to work towards smaller digits.
        
        # How many M's do we need?
        M = num//1000
        for i in range(M):
            numerals.append('M')
        num = num % 1000
        
        # Edge cases of 400 and 900, and any additional D's.
        string = str(num)
        if string[0] == '9' and num >= 900:
            numerals.append('CM')
            num = num % 900
        elif string[0] == '4' and num >= 400:
            numerals.append('CD')
            num = num % 400
        elif num >= 500:
            numerals.append('D')
            num = num % 500
            
        # Remaining hundreds.    
        C = num//100
        num = num % 100
        for i in range(C):
            numerals.append('C')
        
        # Edge cases 90, 40. And any L's.
        string = str(num)
        if string[0] == '9' and num >= 90:
            numerals.append('XC')
            num = num % 90
        elif string[0] == '4' and num >= 40:
            numerals.append('XL')
            num = num % 40
        elif num >= 50:
            numerals.append('L')
            num = num % 50
        
        X = num//10
        num = num % 10
        for i in range(X):
            numerals.append('X')
        
        string = str(num)
        if string[0] == '9' and num >= 9:
            numerals.append('IX')
            num = num % 9
        elif string[0] == '4' and num >= 4:
            numerals.append('IV')
            num = num % 4
        elif num >= 5:
            numerals.append('V')
            num = num % 5

        for i in range(num):
            numerals.append('I')
            
        return ''.join(numerals)

## LeetCode (LC): String to Integer (atoi). https://leetcode.com/problems/string-to-integer-atoi/. Type: String (I did RegEx). Date: 4/22/21.
class Solution:
    def myAtoi(self, s: str) -> int:
        
        import re
        
        # The \d+ assumes that there is at least one digit.
        p = re.compile(r'\s*[+-]?\d+')
        
        m = p.match(s)
        
        if not m:
            return 0
        
        num = int(m.group())
        
        #if num[0] == '+':
        #    num = num[1:]
        #elif num[0] == '-':
        #    num = -int(num[1:])
        
        if num < -2**31:
            num = -2**31
        elif num >= 2**31:
            num = 2**31 - 1
        
        return num

## LeetCode (LC): Longest Palindromic Substring. https://leetcode.com/problems/longest-palindromic-substring/. Type: Strings. Date: 4/22/21.
# O(n^2). This was faster than 87.7% of Python submissions. The code below could probably be cut in half to be made more concise.
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        if len(s) == 0:
            return None
        elif len(s) == 1:
            return s
        
        max_len = 1
        substr = s[0]
        
        for i in range(len(s) - 1):
            
            if s[i] == s[i + 1]:
                distance = min(i, len(s) - i - 2)
                count = self.substring(s, i, 'even', distance)
                if max_len < count:
                    max_len = count
                    substr = s[i - max_len//2 + 1: i + 1 + max_len//2]
            
            distance = min(i, len(s) - i - 1)
            count = self.substring(s, i, 'odd', distance)
            if max_len < count:
                max_len = count
                substr = s[i - max_len//2: i + 1 + max_len//2]
                    
        return substr
        
        
    # Code for a substring.
    def substring(self, s, i, evenodd, distance):
        
        if evenodd == 'even':
            count = 2
        else:
            count = 1
        
        if evenodd == 'even':
            l = i - 1
            r = i + 2
            for j in range(distance):
                if s[l] == s[r]:
                    count += 2
                    l -= 1
                    r += 1
                else:
                    return count
        else:
            l = i - 1
            r = i + 1
            for j in range(distance):
                if s[l] == s[r]:
                    count += 2
                    l -= 1
                    r += 1
                else:
                    return count
        
        return count

            

## LeetCode (LC): Median of Two Sorted Arrays. https://leetcode.com/problems/median-of-two-sorted-arrays/submissions/. Type: Arrays. Date: 4/21/21.
# O(n/2) = O(n).
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        
        from collections import deque
        
        deq1 = deque(nums1)
        deq2 = deque(nums2)
        
        total_len = len(deq1) + len(deq2)
        
        merged = []
        
        for i in range(total_len//2 + 1):
            if not deq1:
                merged.append(deq2.popleft())
            elif not deq2:
                merged.append(deq1.popleft())
            else:
                if deq1[0] <= deq2[0]:
                    merged.append(deq1.popleft())
                else:
                    merged.append(deq2.popleft())
        
        if total_len % 2 == 0:
            return (merged[-1] + merged[-2])/2
        else:
            return merged[-1]

## HR: Circular Array Rotation. https://www.hackerrank.com/challenges/circular-array-rotation/problem. Type: Implementation. Date: 4/21/21.
# O(n)
n, k, q = map(int, input().split())
array = list(map(int, input().split()))

for i in range(q):
    m = int(input())
    print(array[m - k % n])

## HR: Save the Prisoner! https://www.hackerrank.com/challenges/save-the-prisoner/problem. Type: Implementation. Date: 4/21/21.
# O(1)
def saveThePrisoner(n, m, s):
    mod = (s + m-1) % n
    if mod == 0:
        warn = n
    else:
        warn = (s + m-1) % n
    return warn

## HR: Viral Advertising. https://www.hackerrank.com/challenges/strange-advertising/problem. Type: Implementation. Date: 4/17/21.
# O(n).
def viralAdvertising(n):
    likedc = 0
    viewed = 5
    for i in range(1, n + 1):
        likedc += viewed//2
        viewed = (viewed//2) * 3
    return likedc

## HR: Beautiful Days at the Movies. https://www.hackerrank.com/challenges/beautiful-days-at-the-movies/problem. Type: Implementation. Date: 4/17/21.
# O(n)
def beautifulDays(i, j, k):
    count = 0
    for idx in range(i, j + 1):
        s = int(str(idx)[::-1])
        print(s)
        if abs(idx - s) % k == 0:
            count += 1
    return count

## HR: Angry Professor. https://www.hackerrank.com/challenges/angry-professor/problem. Type: Implementation. Date: 4/17/21.
def angryProfessor(k, a):
    ontime = [student for student in a if student <= 0]
    if len(ontime) >= k:
        return 'NO'
    return 'YES'

## HR: Utopian Tree. https://www.hackerrank.com/challenges/utopian-tree/problem. Type: Implementation. Date: 4/17/21.
# O(n)
def utopianTree(n):
    height = 1
    for i in range(1, n+1):
        if i % 2 == 0:
            height += 1
        else:
            height = height * 2
    return height

## HR: Designer PDF Viewer. https://www.hackerrank.com/challenges/designer-pdf-viewer/problem. Type: Implementation. Date: 4/17/21.
# O(n)
def designerPdfViewer(h, word):
    import string
    alph = string.ascii_lowercase
    height = 0
    for letter in word:
        idx = alph.index(letter)
        if h[idx] >= height:
            height = h[idx]
    return height * len(word)

## HR: The Hurdle Race. https://www.hackerrank.com/challenges/the-hurdle-race/problem. Type: Implementation. Date: 4/17/21.
def hurdleRace(k, height):
    m = max(height) - k
    if m <= 0:
        return 0
    else:
        return m

## HR: Picking Numbers. https://www.hackerrank.com/challenges/picking-numbers/problem. Type: Implementation. Date: 4/17/21.
# O(n)
def pickingNumbers(a):
    from collections import Counter
    ac = Counter(a)
    slenmax = 0
    for key in ac:
        slen = max(sum([ac[key-1], ac[key]]), sum([ac[key], ac[key+1]]))
        if slen > slenmax:
            slenmax = slen
    return slenmax

## HR: Cats and a Mouse. https://www.hackerrank.com/challenges/cats-and-a-mouse/problem. Type: Implementation. Date: 4/17/21.
def catAndMouse(x, y, z):
    xz = abs(x - z)
    yz = abs(y - z)
    if xz == yz:
        return 'Mouse C'
    elif xz < yz:
        return 'Cat A'
    else:
        return 'Cat B'

## HR: Electronics Shop. https://www.hackerrank.com/challenges/electronics-shop/problem. Type: Implementation. Date: 4/17/21.
# O(n^2) brute force.
def getMoneySpent(keyboards, drives, b):
    if min(keyboards) + min(drives) > b:
        return -1
    prices = []
    for k in keyboards:
        for d in drives:
            if k + d <= b:
                prices.append(k + d)
    return max(prices)

## HR: Counting Valleys. https://www.hackerrank.com/challenges/counting-valleys/problem. Type: Implementation. Date: 4/17/21.
# O(n)
def countingValleys(steps, path):
    height = 0
    count = 0
    for i in range(steps):
        if path[i] == 'U':
            height += 1
            if height == 0:
                count += 1
        else:
            height -= 1
    return count

## HR: Flipping Bits. https://www.hackerrank.com/challenges/flipping-bits/problem. Type: Bit Manipulation. Date: 4/17/21.
def flippingBits(n):
    s = str(bin(n)[2:])
    zeroes = 32 - len(s)
    s = '0'*zeroes + s
    sl = list(s)
    for i in range(len(sl)):
        if sl[i] == '1':
            sl[i] = '0'
        else:
            sl[i] = '1'
    return int(''.join(sl), 2)

## HR: Sum vs XOR. https://www.hackerrank.com/challenges/sum-vs-xor/problem. Type: Bit Manipulation. Date: 4/17/21.
# This times out on several test cases. It is brute force. 
def sumXor(n):
    count = 0
    for x in range(n+1):
        
        if x + n == n ^ x:
            count += 1 
    return count

## HR: Maximizing XOR. https://www.hackerrank.com/challenges/maximizing-xor/problem. Type: Bit Manipulation. Date: 4/17/21.
# O(n^2)
def maximizingXor(l, r):
    xorlist = []
    for b in range(l, r + 1):
        a = l
        while a <= b:
            xorlist.append(a ^ b)
            a += 1
    return max(xorlist)

## HR: Lonely Integer. https://www.hackerrank.com/challenges/lonely-integer/problem. Type: Bit Manipulation. Date: 4/17/21.
def lonelyinteger(a):
    from collections import Counter
    ac = Counter(a)
    for key in ac:
        if ac[key] == 1:
            return key

## HR: Sherlock and Array. https://www.hackerrank.com/challenges/sherlock-and-array/problem. Type: Sherlock and Array. Date: 4/17/21.
# Second success, inspired by HR discussion.
def balancedSums(arr):
    sumleft = 0
    sumright = sum(arr)
    for i in range(len(arr)):
        sumright -= arr[i]
        if sumleft == sumright:
            return 'YES'
        sumleft += arr[i]
    return 'NO'

# First success. O(n)
def balancedSums(arr):
    if len(arr) == 1:
            return 'YES'
    
    totleft = 0
    totright = sum(arr[1:])
    
    for i in range(len(arr) - 1):
        if totleft == totright:
            return 'YES'
        else:
            totleft += arr[i]
            totright -= arr[i+1]
    if totleft == totright:
        return 'YES'
    return 'NO'

## HR: Missing Numbers. https://www.hackerrank.com/challenges/missing-numbers/problem. Type: Search. Date: 4/17/21.
# O(n).
def missingNumbers(arr, brr):
    from collections import Counter
    a = Counter(arr)
    b = Counter(brr)
    return sorted((b - a).keys())

## HR: Ice Cream Parlor. https://www.hackerrank.com/challenges/icecream-parlor/problem. Type: Search. Date: 4/17/21.
# O(n). Version of Two Sum.
def icecreamParlor(m, arr):
    hashtable = {}
    for idx, price in enumerate(arr):
        diff = m - price
        if diff in hashtable:
            return [hashtable[diff] + 1, idx + 1]
        else:
            hashtable[price] = idx

## HR: Smart Number. https://www.hackerrank.com/challenges/smart-number/problem. Type: Debugging. Date: 4/16/21.
# Kinda odd question. Need to know that perfect squares have an odd number of factors, when duplicates are not included.
# Fix was to check if the number is a perfect square.
import math

def is_smart_number(num):
    val = int(math.sqrt(num))
    if num / val == 1:
        return True
    return False

for _ in range(int(input())):
    num = int(input())
    ans = is_smart_number(num)
    if ans:
        print("YES")
    else:
        print("NO")

## HR: XOR Strings. https://www.hackerrank.com/challenges/strings-xor/problem. Type: Debugging. Date: 4/16/21.
# Corrected: = to ==, res = ... to res +=.
def strings_xor(s, t):
    res = ""
    for i in range(len(s)):
        if s[i] == t[i]:
            res += '0';
        else:
            res += '1';

    return res

s = input()
t = input()
print(strings_xor(s, t))

## HR: Binary Search Tree: Lowest Common Ancestor. https://www.hackerrank.com/challenges/binary-search-tree-lowest-common-ancestor/problem.
## Type: Trees. Date: 4/16/21.
def lca(root, v1, v2):
    v3 = root.info
    current = root
    while True:
        v3 = current.info
        if v1 > v3 and v2 > v3:
            current = current.right
        elif v1 < v3 and v2 < v3:
            current = current.left
        else:
            return current

## HR: Binary Search Tree: Insertion. https://www.hackerrank.com/challenges/binary-search-tree-insertion/problem. Type: Trees. Date: 4/16/21.
    def insert(self, val):
        newnode = Node(val)
        if self.root is None:
            self.root = newnode
        else:
            current = self.root
            while True:
                cval = current.info
                if val <= cval:
                    if current.left:
                        current = current.left
                    else:
                        current.left = newnode
                        break
                if val > cval:
                    if current.right:
                        current = current.right
                    else:
                        current.right = newnode
                        break
        return self.root

## HR: Tree: Level Order Traversal. https://www.hackerrank.com/challenges/tree-level-order-traversal/problem. Type: Trees. Date: 4/16/21.
def levelOrder(root):
    
    from collections import deque
    
    deq = deque([(root, 0, 0)])
    coords = {} 
    
    while deq:
        current, height, distance = deq.popleft() 
        try:
            coords[height].append((distance, current.info))
        except:
            coords[height] = [(distance, current.info)]
        if current.left:
            deq.append((current.left, height + 1, distance - 1))
        if current.right:
            deq.append((current.right, height + 1, distance + 1))
    height_max = height
        
    order = [key for key in coords]
    order.sort()
    
    bfvals = []
    for i in range(height_max + 1):
        tuples = coords[i]
        tuples_sort = sorted(tuples, key=lambda t: (t[0], t[1]))
        vals = [val[1] for val in tuples]
        for v in vals:
            bfvals.append(v)
    
    print(*bfvals)

## HR: Tree: Top View. https://www.hackerrank.com/challenges/tree-top-view/problem. Type: Trees. Date: 4/16/21.
# I have to do a sort here, which is not optimal, but at least the sort is only over unique distances from the center line. Worst case scenario, if the nodes are all
# extended in the same direction, this sort is O(nlogn).
def topView(root):
    # I want to get a tuple of the height and distance from the middle
    # for each node. This can then be compared at the end to print out
    # the top view.
    
    from collections import deque
    
    deq = deque([(root, 0, 0)])
    coords = {}
        
    while deq:
        current, height, distance = deq.popleft()
        try:
            coords[distance].append((height, current.info))
        except:
            coords[distance] = [(height, current.info)]
        if current.left:
            deq.append((current.left, height + 1, distance - 1))
        if current.right:
            deq.append((current.right, height + 1, distance + 1))
        
    order = [key for key in coords]
    order.sort()
    
    topvals = []
    
    for d in order:
        topvals.append((min(coords[d])[1]))
        
    print(*topvals)

## HR: Tree: Height of Binary Tree. https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem. Type: Trees. Date: 4/16/21.
# No recursion.
def height(root):
    from collections import deque    
    deq = deque([(root, 0)])
    while deq:
        current, height = deq.popleft()
        if current.left:
            deq.append((current.left, height + 1))
        if current.right:
            deq.append((current.right, height + 1))
    return height

## HR: Tree: Inorder Traversal. https://www.hackerrank.com/challenges/tree-inorder-traversal/problem. Type: Trees. Date: 4/15/21.
# Without recursion:
def inOrder(root):
    stack = []
    node = root
    
    while stack or node:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            print(node.info, end=' ')
            node = node.right

# Motivated by HR discussion:
def inOrder(root):
    if root:
        inOrder(root.left)
        print(root.info, end=' ')
        inOrder(root.right)

## HR: Tree: Postorder Traversal. https://www.hackerrank.com/challenges/tree-postorder-traversal/problem. Type: Trees. Date: 4/15/21.
# There is a pattern with these solutions. Here is the solution according to that pattern.
def inOrder(root):
    if root:
        inOrder(root.left)
        print(root.info, end=' ')
        inOrder(root.right)

# My first successful attempt: Recursive.
def postOrder(root):
    if root.left:
        postOrder(root.left)
    if root.right:
        postOrder(root.right)
    print(root.info, end=' ')

## HR: Tree: Preorder Traversal. https://www.hackerrank.com/challenges/tree-preorder-traversal/problem. Type: Trees. Date: 4/15/21.
# The standard recursive version:
def preOrder(root):
    if root:
        print(root.info, end=' ')
        preOrder(root.left)
        preOrder(root.right)

# O(n). First successful attempt. This could definitely be better. On the positive side, it prints all elements of a binary tree in preorder without recursion.
def preOrder(root):
    stack = [root]
    current = root
    print(root.info, end=' ')
    while stack:
        if current.left:
            testvisit = current.left
            if testvisit.info != 'visited':
                current = current.left
                print(current.info, end=' ')
                current.info = 'visited'
                stack.append(current)
            elif current.right:                
                testvisit = current.right
                if testvisit.info != 'visited':
                    current = current.right
                    print(current.info, end=' ')
                    current.info = 'visited'
                    stack.append(current)
                else:
                    stack.pop()
                    if not stack:
                        break
                    else:
                        current = stack[-1]
            else:
                stack.pop()
                if not stack:
                    break
                else:
                    current = stack[-1]                
        elif current.right:
            testvisit = current.right
            if testvisit.info != 'visited':
                current = current.right
                print(current.info, end=' ')
                current.info = 'visited'
                stack.append(current)
            else:
                stack.pop()
                if not stack:
                    break
                else:
                    current = stack[-1]
        else:
            stack.pop()
            if not stack:
                break
            else:
                current = stack[-1]

## HR: Reverse a Doubly Linked List. https://www.hackerrank.com/challenges/reverse-a-doubly-linked-list/problem. Type: Linked Lists. Date: 4/14/21.
# O(n).
def reverse(head):
    vals = []
    while head:
        vals.append(head.data)
        head = head.next
    myll = DoublyLinkedList()
    for i in range(len(vals)):
        myll.insert_node(vals[len(vals)-i-1])
    return myll.head

## HR: Inserting a Node into a Sorted Doubly Linked List. https://www.hackerrank.com/challenges/insert-a-node-into-a-sorted-doubly-linked-list/problem.
## Type: Linked Lists. Date: 4/14/21.
# O(n). Janky solution. Can smooth this out.
def sortedInsert(head, data):
    newnode = DoublyLinkedListNode(data)
    if head is None:
        return head
    if data <= head.data:
        newnode.next = head
        return newnode
    node = head
    
    pre = DoublyLinkedListNode(head.data - 1)
    node.prev = pre
    while node:
        if data <= node.data:
            pre = node.prev
            newnode.prev = pre
            newnode.next = node
            node.prev = newnode
            pre.next = newnode
            return head
        else:
            lastnode = node
            node = node.next
    lastnode.next = newnode
    
    return head

## HR: Find Merge Point of Two Lists. https://www.hackerrank.com/challenges/find-the-merge-point-of-two-joined-linked-lists/problem. Type: Linked Lists. Date: 4/14/21.
# I think this is O(n)? Everything is O(n) except there is a search in each set - maybe the scaling of a set search is O(1)?
def findMergeNode(head1, head2):
    if head1 == head2:
        return head1.data
    node1 = head1
    node2 = head2
    ns1 = set()
    ns2 = set()
    while node1 != None or node2 != None:
        pre1 = node1
        pre2 = node2
        if node1 != None:
            ns1.add(node1)
            node1 = node1.next
        if node2 != None:
            ns2.add(node2)
            node2 = node2.next
        if pre1 in ns2:
            return pre1.data
        if pre2 in ns1:
            return pre2.data
    return None

## HR: Delete Duplicate-Value Nodes from a Sorted Linked List. https://www.hackerrank.com/challenges/delete-duplicate-value-nodes-from-a-sorted-linked-list/problem
## Type: Linked Lists. Date: 4/14/21.
# O(n)
def removeDuplicates(head):
    node = head
    vals = [node.data]
    if head is None:
        return None
    while node:
        if node.data != vals[-1]:
            vals.append(node.data)
        else:
            pass
        node = node.next
    myll = SinglyLinkedList()
    for v in vals:
        myll.insert_node(v)
    return myll.head

## HR: Get Node Value. https://www.hackerrank.com/challenges/get-the-value-of-the-node-at-a-specific-position-from-the-tail/problem. Type: Linked Lists. Date: 4/14/21.
# O(n)
def getNode(head, positionFromTail):
    vals = []
    while head:
        vals.append(head.data)
        head = head.next
    return vals[len(vals)-1-positionFromTail]

## HR: Merge Two Sorted Linked Lists. https://www.hackerrank.com/challenges/merge-two-sorted-linked-lists/problem. Type: Linked Lists. Date: 4/14/21.
# Second attempt. Success.
def mergeLists(head1, head2):
    if head1 is None:
        return head2
    if head2 is None:
        return head1
    myll = SinglyLinkedList()
    node1 = head1
    node2 = head2
    while node1 != None or node2 != None:
        if node1 is None:
            myll.insert_node(node2.data)
            node2 = node2.next
            continue
        elif node2 is None:
            myll.insert_node(node1.data)
            node1 = node1.next
            continue
        else:
            if node1.data <= node2.data:
                myll.insert_node(node1.data)
                node1 = node1.next
            else:
                myll.insert_node(node2.data)
                node2 = node2.next
    return myll.head

# First attempt. Times out, and is horrendous. Probably bugged, too.
def mergeLists(head1, head2):
    if head1 is None:
        return head2
    if head2 is None:
        return head1
    node1 = head1
    node2 = head2
    if head1.data <= head2.data:
        head = head1
    else:
        head = head2
    while node1 and node2:
        node1next = node1.next
        node2next = node2.next
        if (node1next is not None) and (node2next is not None):
            if node1.data < node2.data and node2.data < node1next.data:
                node1.next = node2
                node1 = node1next
            elif node2.data < node1.data and node1.data < node2next.data:
                node2.next = node1
                node2 = node2next
        elif (node1next is None) and (node2next is not None):
            if node1.data <= node2.data:
                node1.next = node2
                return head
            else:
                node2 = node2next
        elif (node1next is not None) and (node2next is None):
            if node2.data <= node1.data:
                node2.next = node1
                return head
            else:
                node1 = node1next
        else:
            if node1.data <= node2.data:
                node1.next = node2
                return head
            else:
                node2.next = node1
                return head

## HR: Compare two linked lists. https://www.hackerrank.com/challenges/compare-two-linked-lists/problem. Type: Linked Lists. Date: 4/14/21.
# O(n). A bit more concise, using exclusive or.
def compare_lists(llist1, llist2):
    node1 = llist1
    node2 = llist2
    while node1 and node2:
        if node1.data == node2.data:
            node1 = node1.next
            node2 = node2.next
            if bool(node1) ^ bool(node2):
                return 0
        else:
            return 0
    return 1

# Original.
def compare_lists(llist1, llist2):
    node1 = llist1
    node2 = llist2
    while node1 and node2:
        if node1.data == node2.data:
            node1 = node1.next
            node2 = node2.next
            if (node1 is None) and (node2 is not None):
                return 0
            elif (node1 is not None) and (node2 is None):
                return 0
        else:
            return 0
    return 1

## HR: Reverse a Linked List. https://www.hackerrank.com/challenges/reverse-a-linked-list/problem. Type: Linked Lists. Date: 4/14/21.
# O(n). I get the list of data values from the original linked list, then construct a new linked list from this, in reverse.
def reverse(head):
    if not head:
        return Null
    else:
        nodevals = []
        node = head
        while node:
            nodevals.append(node.data)
            node = node.next
    myllist = SinglyLinkedList()
    for i in range(len(nodevals)):
        myllist.insert_node(nodevals[-i - 1])
    return myllist.head

## HR: Print in Reverse. https://www.hackerrank.com/challenges/print-the-elements-of-a-linked-list-in-reverse/problem. Type: Linked Lists. Date: 4/14/21.
# O(n). Could be more efficient by avoiding reverse(). Could use deque.
def reversePrint(head):
    if not head:
        pass
    else:
        llist = []
        node = head
        while node:
            llist.append(node.data)
            node = node.next
        llist.reverse()
        print(*llist, sep='\n')

## HR: Delete a Node. https://www.hackerrank.com/challenges/delete-a-node-from-a-linked-list/problem. Type: Linked Lists. Date: 4/14/21.
# O(n^2)
def deleteNode(head, position):
    if position == 0:
        post = head.next
        del(head)
        return post
    else:
        node = head
        for i in range(position-1):
            node = node.next
        pre = node
        delnode = node.next
        post = delnode.next
        pre.next = post
        del(delnode)
        return head

## HR: Insert a node at a specific position in a Linked List. 
## https://www.hackerrank.com/challenges/insert-a-node-at-a-specific-position-in-a-linked-list/problem. Type: Linked Lists. Date: 4/14/21. 
def insertNodeAtPosition(head, data, position):
    if not head:
        return SinglyLinkedListNode(data)
    elif position == 0:
        newhead = SinglyLinkedListNode(data)
        newhead.next = head
        return newhead
    else:
        node = head
        for i in range(position-1):
            node = node.next
        pre = node
        post = node.next
        newnode = SinglyLinkedListNode(data)
        pre.next = newnode
        newnode.next = post
        return head

## HR: Insert a node at the head of a Linked List. https://www.hackerrank.com/challenges/insert-a-node-at-the-head-of-a-linked-list/problem. Type: Linked Lists. Date: 4/14/21.
def insertNodeAtHead(llist, data):
    if not llist:
        head = SinglyLinkedListNode(data)
        return head
    else:
        oldhead = llist
        newhead = SinglyLinkedListNode(data)
        newhead.next = oldhead
        return newhead

## HR: Insert a Node at the Tail of a Linked List. https://www.hackerrank.com/challenges/insert-a-node-at-the-tail-of-a-linked-list/problem. Type: Linked Lists. Date: 4/14/21.
def insertNodeAtTail(head, data):
    if not head:
        head = SinglyLinkedListNode(data)
        return head
    node = head
    while node.next:
        node = node.next
    node.next = SinglyLinkedListNode(data)
    return head

## HR: Print the Elements of a Linked List. https://www.hackerrank.com/challenges/print-the-elements-of-a-linked-list/problem. Type: Linked Lists. Date: 4/14/21.
def printLinkedList(head):
    node = head
    while True:
        print(node.data)
        node = node.next
        try:
            node.data
        except:
            break

## HR: Equal Stacks. https://www.hackerrank.com/challenges/equal-stacks/problem. Type: Stacks. Date: 4/13/21.
from collections import deque

n1, n2, n3 = list(map(int, input().split()))

s1 = deque(map(int, input().split()))
s2 = deque(map(int, input().split()))
s3 = deque(map(int, input().split()))

h1 = sum(s1)
h2 = sum(s2)
h3 = sum(s3)

while h1 != h2 or h1 != h3:
    if h1 > h2:
        h1 -= s1.popleft()
    elif h1 < h2:
        h2 -= s2.popleft()
    if h1 > h3:
        h1 -= s1.popleft()
    elif h1 < h3:
        h3 -= s3.popleft()
        
print(h1)

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
