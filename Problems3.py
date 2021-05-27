## LC: Find Anagram Mappings.
# O(n^2)
class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        
        mapping = []
        
        for n in nums1:
            mapping.append(nums2.index(n))
        
        return mapping
