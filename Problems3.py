## LC: Find Anagram Mappings. https://leetcode.com/problems/find-anagram-mappings/. Type: Arrays. Date: 5/26/21.
# O(n^2)
class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        
        mapping = []
        
        for n in nums1:
            mapping.append(nums2.index(n))
        
        return mapping
