# # Guards in a Museum
# 
# A museum is represented by a matrix of size n, rows, by m columns.  Each square has either an "o" for an open space, a "w" for a wall, or a "g" for a guard.  For a given museum matrix, return another matrix where all of the "o" spaces are replaced with their manhattan distance from the nearest guard.
# 
# Solution:
#     
# Walk though the matrix, O(n * m) or O(N) where N is number of entries in the matrix, and keep track of the guard locations.
# Then iterate through the guards, and do a BFS outward, and count the distance as you walk.  Stop the BFS from each guard when you reach an already filled in space where the entry is <= the current distance.  The run-time is O(GxN), where G is the number of guards.  Space complexity should be O(N + G), because we can replace the "o"'s in place, and we need a stack for the guards.  In the case we don't want to modify the original matrix, it's O(2N + G) space.
# 

m = [
    ["O", "O", "O", "O", "O", "O", "O", "W", "O", "G"],
    ["O", "O", "O", "O", "O", "O", "O", "W", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "W", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "W", "W", "W"],
    ["O", "O", "O", "O", "W", "O", "G", "O", "O", "O"],
    ["W", "W", "W", "O", "W", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
    ["G", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
]


def make_guard_distance_map(m):
    n_rows = len(m)
    n_cols = len(m[0])
    
    def valid_next_space(row, col, dist):
        if (-1 < row < n_rows) and (-1 < col < n_cols):
            if not m[row][col] == "G" and not m[row][col] == "W":
                if m[row][col] == "O":
                    return True
                if m[row][col] > dist:
                    return True
        return False
    
    def find_guards():
        guards = []
        for row in range(n_rows):
            for col in range(n_cols):
                if m[row][col] == "G":
                    guards.append((row, col, 0))
        return guards
    
    
    def bfs_step(this_step):
        n_this = len(this_step)
        next_step = []
        for i in range(n_this):
            row, col, dist = this_step.pop()
            # up
            if valid_next_space(row + 1, col, dist + 1):
                m[row + 1][col] = dist + 1
                next_step.append((row + 1, col, dist + 1))
            # down
            if valid_next_space(row - 1, col, dist + 1):
                m[row - 1][col] = dist + 1
                next_step.append((row - 1, col, dist + 1))
            # left
            if valid_next_space(row, col - 1, dist + 1):
                m[row][col - 1] = dist + 1
                next_step.append((row, col - 1, dist + 1))
            # right
            if valid_next_space(row, col + 1, dist + 1):
                m[row][col + 1] = dist + 1
                next_step.append((row, col + 1, dist + 1))

        if next_step:
            bfs_step(next_step)
    
    guards = find_guards()
    
    for g in guards:
        bfs_step([g])


make_guard_distance_map(m)
expected = [
    [10 ,   9,   8,  7,   6,  5,  4,'W',   1, 'G'], 
    [9  ,   8,   7,  6,   5,  4,  3,'W',   2,   1], 
    [10 ,   9,   8,  7, 'W',  3,  2,'W',   3,   2], 
    [11 ,  10,   9,  8, 'W',  2,  1,'W', 'W', 'W'], 
    [11 ,  10,   9,  8, 'W',  1,'G',  1,   2,   3], 
    ['W', 'W', 'W',  7, 'W',  2,  1,  2,   3,   4], 
    [  3,   4,   5,  6, 'W',  3,  2,  3,   4,   5], 
    [  2,   3,   4,  5, 'W',  4,  3,  4,   5,   6], 
    [  1,   2,   3,  4, 'W',  5,  4,  5,   6,   7], 
    ['G',   1,   2,  3, 'W',  6,  5,  6,   7,   8]
]
print(m == expected)








# # Outline
# 
# An exploration strategy I have proposed involves choosing an order for scored campaigns by sampling from the distribution of predicted reward for the candidate advertising campaigns-targets.  
# 
# “Reward” is a term often used to describe the benefit from an action in the mutli-armed bandit problem.  In our case, “predicted reward” is referring to the algorithm predicted eCPM for a advertising campaigns-target pair.  
# 
# In words, the procedure is to treat the set of eCPMs as the elements of a multinomial, and sample from that distribution without replacement. By doing so, we will generate a new, randomized, order for the scored, candidate campaign-targets, where the expected value of the distributions of the relative elements, will retain the same order as the prediced reward.
# 

# # The algorithm
# 
# In pseudo-code, the inefficient, but direct form of the algorithm is as follows:
# 

"""

alpha = a parameter used to make the algorithm more or less greedy (higher is more greedy)
N = number of candidate campaigns
L = list of scored, candidate campaigns
O = output, re-ordered, scored, candidate campaigns

scaledL = for each campaign, raise the eCPM to eCPM ^ alpha

normL = normalize the list of scaledL (where we devide each eCPM ^ alpha by the sum of them all)

cumNormL = perform a cumulative sum from the start to the end of the 
            normalized list of scored, candidate campaigns 
            (the first campaigns cumEcpm will be 0, while the last will be 1 - it's normed ecpm)

while (N > 0) do
    compute normL from current elements of L
    compute cumNorm from normL
    i = rand(0,1)
    for j in range 0 to length of cumNormL:
        if i > cumNormL(j):
            append cumNormL(j) to the tail of O
            remove the jth element from L
            N -= 1
            break
        else:
            continue
"""


# # A more efficient, memoized approach
# 
# In practice, we don't have to re-normalize the distribution every time we sample.  We can When streams or generators are suppored, that would also be preferable.
# 

from collections import namedtuple
from random import random as rand

Campaign = namedtuple('Campaign', ['id', 'ecpm'])

class Sampler:
    
    def __init__(self, campaigns, alpha = 1.0):
        self.remaining_campaigns = {}
        self.total_sum = 0
        for c in campaigns:
            scaled_ecpm = c.ecpm ** alpha
            self.ramaining_campaigns[c.id] = (scaled_ecpm, c)
            self.total_sum += scaled_ecpm
    
    def size(self):
        return len(self.ramaining_campaigns)
    
    def rand(self):
        return rand() * self.total_sum
    
    def get_next_sample_id(self):
        lastCumSum = 0.0
        r = this.rand()
        for cid, t in self.ramaining_campaigns.items():
            next_cum_sum = lastCumSum + t[0]
            if next_cum_sum >= r:
                return cid
            else:
                lastCumSum = next_cum_sum
        
    def pop(self, cid):
        self.totalSum -= self.remaining_campaigns[cid][0]
        return self.remaining_campaigns.pop(cid)
    
    def get_next_sample(self):
        if self.size() > 0:
            next_cid = get_next_sample_id()
            return pop(next_cid)
        else:
            return None





# ## Group odd and even elements of a linked list
# 
# Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.
# 
# You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.
# 
# Example:
# Given 1->2->3->4->5->NULL,
# return 1->3->5->2->4->NULL.
# 
# Note:
# The relative order inside both the even and odd groups should remain as it was in the input. 
# The first node is considered odd, the second node even and so on ...
# 

# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def oddEvenList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    odd_head = None
    odd_tail = None
    even_head = None
    even_tail = None
    while not (head==None):
        if head.val % 2 == 0:
            if even_tail == None:
                even_tail = head
                even_head = head
            else:
                even_tail.next = head
                even_tail = head
        else:
            if odd_tail == None:
                odd_tail = head
                odd_head = head
            else:
                odd_tail.next = head
                odd_tail = head

        head = head.next

    odd_tail.next = even_head
    return odd_head





# # Sub matrix sum
# 
# Given a 2D array (matrix), define a method that efficiently computes the sum of the sub-matrix indexed by x_1, y_1 in the upper left, and x_2, y_2 in the lower right
# 

# This is solved most efficiently at query time by doing some pre-processing.  
# When the method is configured, we compute a Summed-Area-Table.

def getSafeMatValue(m, x, y):
    r = len(m)
    c = len(m[0])
    if y >= 0 and y < r and x >=0 and x < c:
        return m[y][x]
    else:
        return 0
        
def buildSAT(m):
    r = len(m)
    c = len(m[0])
    SAT = [[0 for x in range(c)] for y in range(r)]
    for x in range(c):
        for y in range(r):
            SAT[y][x] = getSafeMatValue(m, x, y)
            + getSafeMatValue(SAT,x-1,y) 
            + getSafeMatValue(SAT,x,y-1) 
            - getSafeMatValue(SAT,x-1,y-1)
    return SAT

def config_sum_matrix_sum(m):
    SAT = buildSAT(m)
    def sumMatrixSum(x1, y1, x2, y2):
        s = getSafeMatValue(SAT, x2, y2)
        + getSafeMatValue(SAT, x1, y1)
        - getSafeMatValue(SAT,y1-1,x2) 
        - getSafeMatValue(SAT,y2,x1-1)
        return s
    return sumMatrixSum





# # Min sub-array
# 
# Given an array of numbers of size N, return the continuous, sub-array of size q with the minimum sum of the sub-array.
# 

def getMinSubArray(A, q):
    if q >= len(A):
        return sum(A)
    
    n = len(A)
    minArray = A[:q]
    lastArray = A[:q]
    lastSum = sum(minArray)
    minSum = sum(minArray)
    
    for i in range(1, n - q + 1):
        rightIdx = i + q
        newSum = lastSum + A[rightIdx] - lastArray[0]
        newSubArray = lastArray[1:].append(A[rightIdx])
        if (newSum < minSum):
            minArray = nextSubArray
            minSum = newSum
        
        lastSum = newSum
        lastArray = newSubArray
    
    return (lastSum, lastArray)
        





# # MAP Abstract Data Type
# 

# - M[key] : returns value stored at 'key'
# - M[key] = value : insert 'key' that points to 'value'
# - del M[key] : remove 'key' from the set of keys and any value that to which it points
# - len(M) : return the number of keys in the map
# - iter(M) : iterate over the sequence of keys, if this is not an ordered map, then ordering is not guarenteed
# 
# - M.get(k) : same as M[key]
# - M.setdefault(key, default_v) : get the 'value' at 'key', else return the 'default_v'
# - M.pop(key, default_v) : same as M.setdefault, but remove 'key' from the map
# - M.keys() : return an iterable of the set of keys in the map
# - M.values() : return an iterable of the set of values in the map
# - M.update(M2) : for every (key -> value) in M2, either insert or replace the current (key -> value) in M
# - M.clear() : remove all (key -> value) from M
# 

# ## inheriting from mutableMap
# 

from collections import MutableMapping

class MapBase(MutableMapping):
    class Item:
        __slots__ = '_key', '_value'
        
        def __init__(self,k,v):
            self._key = k
            self._value = v
            
            





# Design a deck of cards, a standard deck, to be used for various games


from random import randint

class Card:
    
    def __init__(self, value, suit):
        """
        value is the cards number or type as a string, ace, 2, 3, ..., jack, queen, king
        """
        # check valid values and suits
        self.suit = suit
        self.value = value

    def get_order_value(self, value, ace_high):
        """
        a method to get a numeric value for the cards face
        """
        if value in "23456789" or value == "10":
            order_val = int(value)
        elif value == "jack":
            order_val = 11
        elif value == "queen":
            order_val = 12
        elif value == "king":
            order_val = 13
        elif value == "joker":
            order_val = -1
        elif ace_high:
            order_val = 14
        else:
            order_val = 1
            
        return order_val

class Deck_of_cards:
    
    values = [str(i) for i in range(2,11)] + ["jack", "queen", "king", "ace"]
    suits = ["hearts", "diamonds", "spades", "clubs"]
    jokers = [Card("joker", None), Card("joker", None)]
    standard_deck_no_jokers = [] # standard deck of cards
    
    def __init__(self, num_decks = 1, jokers = False):
        self.num_decks = num_decks
        self.jokers = jokers

    def initialize(self):
        if self.jokers:
            one_deck = standard_deck_no_jokers + jokers
        else:
            one_deck = standard_deck_no_jokers
        
        self.stack = one_deck*self.num_decks # the cards remaining in the deck, in a stack
        self.shuffle()
    
        
    def remaining(self):
        return len(self.stack)
    
    
    def collect_cards(self):
        """
        re-initialize the deck with all cards
        """
        self.stack = one_deck*num_decks
        
    
    def shuffle(self):
        """
        randomly re-order the stack
        """
        n = len(self.stack)
        for i in range(n-1):
            j = randint(i,n-1)
            self.stack[i], self.stack[j] = self.stack[j], self.stack[i]
    
    
    def peek(self):
        return self.stack[0]
    
    
    def draw(self, num = 1):
        return [self.stack.pop(0) for i in range(num)]
    
    
class CardGame:
    
    def __init__(self):
        self.deck = Deck_of_cards()
        
    # game specific behavior


# # Memcache for large files
# 
# Design a cache for large objects (> 1MB) on top of memcache.
# 
# You have access to the following memcache methods:
#     
#     memcache.get(key)
#     memcache.set(key, value)
#     
# 

# The interface for the BigCache should be similarly simple from the users perspective.  We should have the basic get and get and set operations, and additionally a delete operation:
# 
#     BigCache.get(key)
#     BigCache.set(key, value)
#     BigCache.delete(key)
# 
# Underneath, we should partition our memcache into blocks of the maximum allowed size, 1MB, emumerate them, then regulate access with a set.  So, if we have a memcache of 1GB, we have 1000 available blocks, from 0 to 999.  
# 
# We'll need to mainain an arrays of available blocks, and a map from key to used blocks associated with that key.
# 
# So, as a class it might look like this:
# 

class obj:
    
    def __init__(self,value,size):
        self.value = value
        self.size = size # in MB



class memcache:
    
    def __init__(self, size):
        self.size = size # in MB
        self.cache = {}
    
    def get(self, key):
        return self.cache[key]
    
    def set(self, key, value):
        self.cache[key] = value


class BigCache:
    
    def __init__(self, mem):
        blocks = mem.size
        self.avaiable_blocks = list(range(0, blocks))
        self.mem = mem
        self.cache = {} # map from the object key, to the block inices
    
    def get(self, key):
        if key in self.cache:
            partitions = [self.mem.get(idx) for idx in self.cache[key]]
            return _reconstruct_object(partitions)
        else:
            raise Exception('Key not found')
        
        
    
    def set(self, key, value):
        if len(self.avaiable_blocks) >= _get_necessary_blocks(value):
            self.cache[key] = []
            partitioned = _partition(value)
            for i in partitioned:
                idx = self.avaiable_blocks.pop()
                self.cache[key].append(idx)
                self.mem[idx] = i
        else:
            raise Exception('cache is full')
        
    
    def delete(self, key):
        for idx in self.cache[key]:
            # set blocks associated with this key to available
            # we might want to explicitly remove them, but our memcache api doesn't allow this
            self.avaiable_blocks.append(idx)
    
    def _get_necessary_blocks(self,value):
        blocks = value.size//1
        if (value.size % 1) > 0:
            blocks += 1
        return blocks
    
    def _partition(self, value):
        remainder = value.size % 1
        return [obj(value.value,1) for i in range(value.size//1)] + [(obj(v,remainder))]
    
    def _reconstruct_object(self, partitions):
        size = len(partitions) + partitions[-1].size
        return obj(partitions[0].value, size)


# # A messager system backend
# 
# ### Application API: 
# 
# We need an application layer API.  This should be the entrypoint for all reading and writing operations.
# 
# ### Distributed application server logic:
# 
# To make the system highly available, we need a distributed, resiliant cluster of servers to perform the logic underlaying the reads and writes the API makes available.  This system should be supported by some cluster management system like zookeeper, which manages our nodes and balances and distributes the work.
# 
# ### Application buisness logic
# 
# This is where actuall application logic should live.  This is where we have our logic about how to make messages available, what to get when users are getting messages, and what to set when new messages are sent.  
# 
# We should have write-though caching here to make recent writes available quickly.  
# 
# ### Data access layer
# 
# This is where we store the data schema, and how to access and update the message data.  The underlaying data will essentially be timestamped, multi-user-id tagged, messages.  The schema will essentially look like:
#     
#     (timestamp, [user_ids], message) 
#     which is of type (timestamp (int,bigint,time), list of strings, message object (a string or an enhanced string that supports emoji type stuff))
# 
# The underlaying store should be pluggable, so we can swap stores as necessary.
# 




# # A naive implimentation of a Least Recently Used Cache
# 

class LRUCacheNaive:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tm = 0
        self.cache = {}
        self.lru = {}

    def get(self, key):
        if key in self.cache:
            self.lru[key] = self.tm
            self.tm += 1
            return self.cache[key]
        return -1

    def set(self, key, value):
        if len(self.cache) >= self.capacity:
            # find the LRU entry
            # this is an O(n) search for the minimum lookup value
            old_key = min(self.lru.keys(), key=lambda k:self.lru[k])
            self.cache.pop(old_key)
            self.lru.pop(old_key)
        self.cache[key] = value
        self.lru[key] = self.tm
        self.tm += 1


# # A constant time implimentation using python's OrderedDict
# 

import collections

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        # ordered dict
        self.cache = collections.OrderedDict()

    def get(self, key):
        # we re-insert the item after each lookup, thus the ordered dict is ordering items by their last access
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def set(self, key, value):
        try:
            # if in the cache, remove it
            self.cache.pop(key)
        except KeyError:
            # if we're over capacity, remove the oldest item
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        # re-insert it
        self.cache[key] = value


# # Using python base types
# 

class Node:
    def __init__(self, key, value, prev = None, next = None):
        self.key = key
        self.value = value
        self.next = next
        self.prev = prev

    def pop_self(self):
        self.prev.next = self.next
        self.next.prev = self.prev
        return self.key


class DoubleLinkedList:

    def __init__(self):
        # explicit head and tail nodes
        self.tail = Node("tail", None, None, None)
        self.head = Node("head", None, prev = None, next = self.tail)
        self.tail.prev = self.head

    def insert(self, new_node):
        new_node.prev = self.tail.prev
        new_node.next = self.tail
        self.tail.prev.next = new_node # old last node points to new
        self.tail.prev = new_node # old tail points to newest node

    def pop_oldest(self):
        return self.head.next.pop_self()


class LRUCache:
    
    def __init__(self, capacity = 10):
        self.capacity = capacity
        self.size = 0
        self.cache = {}
        self.order = DoubleLinkedList()

    
    def get(self, key):
        if key in self.cache():
            # if in cache, remove from queue and re-instert at end
            node = self.cache[key]
            node.pop_self()
            self.order.insert(node)
            return node.value
        else:
            return None
    
    
    def set(self, key, value):
        if (self.size + 1) > self.capacity:
            oldest_key = self.order.pop_oldest() # remove oldest form order
            self.cache.pop(oldest_key) # remove oldest from dict
        else:
            self.size += 1
        new_node = Node(key, value, None, None)
        self.cache[key] = new_node # insert new into dict
        self.order.insert(new_node) # insert new into end of ordering
    
    
    def remove(self, key):
        if key in self.cache:
            node = self.cache.pop(key)
            node.pop_self()
            self.size -= 1
            return node.value
        else:
            return None





# ## powers of 3
# 

import math
def is_power_of_three(n):
    """
    :type n: int
    :rtype: bool
    """
    if n < 1:
        return False
    if n == 1:
        return True
    else:
        while n > 3:
            n = n / float(3)

        if n == 3:
            return True
        else:
            return False

print is_power_of_three(1) == True
print is_power_of_three(2) == False
print is_power_of_three(3) == True
print is_power_of_three(6) == False
print is_power_of_three(9) == True
print is_power_of_three(27) == True
print is_power_of_three(28) == False
print is_power_of_three(81) == True
print is_power_of_three(100) == False
print is_power_of_three(81*3) == True


## max number


# Given two arrays of length m and n with digits 0-9 representing two numbers. Create the maximum number of length k <= m + n from digits of the two. The relative order of the digits from the same array must be preserved. Return an array of the k digits. You should try to optimize your time and space complexity.
# 
# Example 1:
# 
# nums1 = [3, 4, 6, 5]
# 
# nums2 = [9, 1, 2, 5, 8, 3]
# 
# k = 5
# 
# return [9, 8, 6, 5, 3]
# 
# Example 2:
# 
# nums1 = [6, 7]
# 
# nums2 = [6, 0, 4]
# 
# k = 5
# 
# return [6, 7, 6, 0, 4]
# 
# Example 3:
# 
# nums1 = [3, 9]
# 
# nums2 = [8, 9]
# 
# k = 3
# 
# return [9, 8, 9]
# 




import numpy as np
from copy import deepcopy


# # Find maximum sum of sub-arrays
# 
# Given an array, of alln(n+1)/2 sub-arrays, find the max_i(sum(sub_array_i))
# 

def find_max_sub_array_sum_only(a):
    if a == None: 
        return None
    l = len(a)
    if l < 2: 
        return sum(l)
    cur_sum = 0
    max_sum = 0
    for idx,val in enumerate(a):
        if cur_sum <= 0:
            cur_sum = val
        else:
            cur_sum += val
        
        if cur_sum > max_sum:
            max_sum = cur_sum
        
    return max_sum

def find_max_sub_array(a):
    if a == None: 
        return None
    l = len(a)
    if l < 2: 
        return sum(l)
    cur_sum, max_sum = (0, 0)
    cur_array, max_array = ([], [])
    for idx,val in enumerate(a):
        if cur_sum <= 0:
            cur_sum = val
            cur_array = [idx]
        else:
            cur_sum += val
            cur_array.append(idx)
        
        if cur_sum > max_sum:
            max_sum = deepcopy(cur_sum)
            max_array = deepcopy(cur_array)
    
    return [a[i] for i in max_array]


a = [1, -2, 3, 10, -4, 7, 2, -5]
max_sub_array = [3, 10, -4, 7, 2]

print find_max_sub_array(a) == max_sub_array
print find_max_sub_array(a)


# # Count the Number of Paths  <=  a max-cost
# 
# Given a matrix, with a cost placed in each cell, count the number of paths from the 
# top-left to the bottom-right without the sum of the path exceeding some max-cost (k)
# 

# run


def count_path_rec(mat, m, n, k):
    # base case
    if (m < 0) or (n < 0):
        return 0
    elif (m==n==0) and (k >= mat[m][n]): 
        return 1
    else:
        return count_path_rec(mat, m-1, n, k - mat[m][n]) + count_path_rec(mat, m, n-1, k - mat[m][n])

def path_count(mat, k):
    m = len(mat) - 1
    n = len(mat[0]) - 1
    if m == n == 0:
        return 1 if mat[0][0] == k else 0

    return count_path_rec(mat, m, n, k)

mat = [[1, 2, 3],
       [4, 6, 5],
       [3, 2, 1]]

tests = [
    path_count(mat,10) == 0,
    path_count(mat,11) == 1,
    path_count(mat,12) == 3,
    path_count(mat,13) == 3,
    path_count(mat,14) == 4,
    path_count(mat,15) == 5,
    path_count(mat,16) == 5,
    path_count(mat,17) == 6 ]

print all(tests)



# # Find largest sub-matrix
# 
# In a binary matrix, find the largest sub-matrix of all 1's
# 

def find_max_pos_sub_mat(M):
    R = len(mat)
    C = len(mat[0])
    S = deepcopy(mat)
    
    maximum = 0
    for r in range(1,R):
        for c in range(1,C):
            # increase the counter if current is one, 
            # left, above, and left_above coutners are all non-zero
            if M[r][c]:
                S[r][c] = min(S[r][c-1], S[r-1][c], S[r-1][c-1]) + 1
            else:
                S[r][c] = 0
            
            if maximum < S[r][c]:
                maximum = S[r][c]
                max_pos = (r,c)

    top_left = 9
    return (S, maximum, max_pos)


mat =  [[0,1,1,0,1],
        [1,1,0,1,0],
        [0,1,1,1,0],
        [1,1,1,1,0],
        [1,1,1,1,1],
        [0,0,0,0,0]]

# solution: (2,1) to (4,3)
S, maximum, max_pos = find_max_pos_sub_mat(mat)   
print np.array(S)
print maximum
print max_pos


# ## Build the power set
# 


def build_power_set(in_set):
    out = [[]] # start with empty set
    for i in in_set:
        new_sets = []
        for previous_set in out:
            new_sets.append(deepcopy(previous_set) + [i])
        out.extend(new_sets)
    return out

s = list(range(3))
ps = build_power_set(s)
answer = [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
print ps == answer





get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn
from sklearn.datasets import make_blobs

# setup pyspark for IPython_notebooks
spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))


data_home = os.environ.get('DATA_HOME', None)
sys.path.insert(0, data_home)

# data
from gen_data import make_blobs_rdd

# utilitiy functions for this notebook
from lsh_util import *

# make some data
N = 1000
d = 2
k = 3
sigma = 1
bound = 10*sigma

data_RDD = make_blobs_rdd(N, d, k, sigma, bound, sc)
data_RDD.take(2)


# # Stable Distribution Projection Hashing
# 

def config_stable_dist_proj(d, p = 5, r = 2.0, seed = None):
    # random projection vectors
    A = np.random.multivariate_normal(np.zeros(d), np.eye(d), p)
    B = np.random.rand(1,p)
    def projection_func(tup):
        y, x = tup # expect key (int, 1xD vector)
        projs = ((A.dot(x) / r) + B).flatten()
        bucket = to_bucket(projs)
        
        return (bucket, y)
    
    return (A, B, projection_func)


A, B, hash_func = config_stable_dist_proj(d)

gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
for b, g, c in sorted(gini_impurities):
    print "bucket: %s , in bucket: %d , gini_impurity: %f" % (b, c, g)


# impurity as we scale up the number of hyperplanes used for projections
for n_Z in range(10,201,10):
    A, B, hash_func = config_stable_dist_proj(d, n_Z)
    gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
    g_i = weighted_gini(gini_impurities)
    print "%d projections, gini_impurity: %f" % (n_Z, g_i)
    





# # Min (or max, not min-max) queue
# 
# Using an array
# 
# exposed API:
# initialized with:
# 
#     queue = minQueue()
#     
# exposed methods:
# 
#     queue.size() # the size of the queue
#     
#     queue.pop() # remove the minimum element and update the queue
#     
#     queue.push(item) # insert new item into the queue
# 

class MinQueue:
    
    def __init__(self):
        self.queue = [None]
        self.size = 0

    def _getParentIdx(self, idx):
        return idx / 2
    
    def lookParent(self, idx):
        return _lookIdx(_getParentIdx(idx))
    
    def _getLeftChildIdx(self, idx):
        return idx * 2
    
    def _lookLeftChild(self, idx):
        return _lookIdx(self._getLeftChildIdx(idx))
    
    def _getRightChildIdx(self, idx):
        return idx * 2 + 1
    
    def _lookRightChild(self, idx):
        return _lookIdx(self._getRightChildIdx(idx))
    
    def _lookIdx(self, idx):
        return self.queue[idx]
    
    def _swap(a,b):
        self.queue[a], self.queue[b] = self.queue[b], self.queue[a]
    
    def _nodeExists(self, idx):
        return idx > 0 and idx <= self.size()
    
    def _shiftUp(self, idx):
        if (_nodeExists(_getParentIdx(idx))) and (lookParent(idx) < _lookIdx(idx)):
            _swap(_getParentIdx(idx), idx)
            _shiftUp(_getParentIdx(idx))
        else:
            return
    
    def _shiftDown(self, idx):
        if not _nodeExists(_getLeftChildIdx(idx)):
            return
        elif not _nodeExists(_getRightChildIdx(idx)):
            if (_lookIdx(idx) > _lookLeftChild(idx)):
                _swap(_getLeftChildIdx(idx), idx)
        
        elif (_lookIdx(idx) > _lookLeftChild(idx)) or (_lookIdx(idx) > _lookRightChild(idx)):
            if (_lookRightChild(idx) > _lookLeftChild(idx)):
                _swap(_getRightChildIdx(idx), idx)
                _shiftDown(self, _getRightChildIdx(idx))
            else:
                _swap(_getLeftChildIdx(idx), idx)
                _shiftDown(self, _getLeftChildIdx(idx))
        else:
            return
    
    def size(self):
        return size
    
    def pop(self):
        minVal = self.queue[1]
        self.queue[1] = self.queue[-1]
        del self.queue[-1]
        self.size -= 1
        _shiftDown(1)
        return minVal
    
    def push(self, item):
        self.queue.append(item)
        self.size += 1
        _shiftUp(self.size)

    
    def look(self):
        return self.queue[0]





# # Latent Dirichlet Allocation on Web-App events
# 
# LDA is a clustering model often applied to topic modeling.  Roughly LDA uses a graphical model of nested multinomials to destribe the distribution of tokens (words) within topics, and topics within documents.  
# 
# Gaphics for the model's mathematics is left as a TODO.  For now [LDA Wikipedia Article](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation).
# 
# However, this nested multinomial structure can be applied to other settings as well.  In this case, if we are interested in clustering behavior withing a web-application. If we are minitoring unique events within the application, and we assume that users of the application come to the app with some underlaying purpose for each session (uninterupted period of use) in the application (the cardinality of total purposes for sessions being much smaller than that of events), we can cluster sesssions by event frequency the same way we would cluster documents by word frequency.  Structurally, the session-event model can be equivilent to the topic-word model.
# 

# ### Sessions
# 
# For this experiment, events were sessionized by glomming together events that occured within 90min of each other; except sessions were not allowed to exceed 24hours in length.
# 

# Online-LDA

import sys, re, time, string
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def parse_sessions_list(sessions, event_set):

    D = len(sessions)
    
    eventsids = list()
    eventscts = list()
    for D in range(0, D):
        events = sessions[D]
        ddict = dict()
        for e in events:
            if (e in event_set):
                eventtoken = event_set[e]
                if (not eventtoken in ddict):
                    ddict[eventtoken] = 0
                ddict[eventtoken] += 1
        eventsids.append(ddict.keys())
        eventscts.append(ddict.values())

    return((eventsids, eventscts))

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, event_set, K, D, alpha = None, eta = None, tau0 = 1024, kappa = 0.7):
        self._events = dict()
        for events in event_set:
            events = events.lower()
            self._events[events] = len(self._events)

        self._K = K
        self._W = len(self._events)
        self._D = D
        self._alpha = alpha if alpha else 1.0 / K
        self._eta = eta if eta else 1.0 / K
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1 * n.random.gamma(100.0, 1.0 / 100.0, (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def update_lambda(self, sessions):

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(sessions)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(sessions, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) +             rhot * (self._eta + self._D * sstats / len(sessions))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, sessions, gamma):
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(sessions).__name__ == 'string'):
            temp = list()
            temp.append(sessions)
            sessions = temp

        (eventsids, eventscts) = parse_sessions_list(sessions, self._events)
        batchD = len(sessions)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(sessions | theta, id)]
        for d in range(0, batchD):
            gammad = gamma[d, :]
            ids = eventsids[d]
            cts = n.array(eventscts[d])
            phinorm = n.zeros(len(ids))

            for i in range(0, len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(sessions)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(model._eta*model._W) - 
                              gammaln(n.sum(model._lambda, 1)))

        return (score)


    def do_e_step(self, sessions):
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(sessions).__name__ == 'string'):
            temp = list()
            temp.append(sessions)
            sessions = temp

        (eventsids, eventscts) = parse_sessions_list(sessions, self._events)
        batchD = len(sessions)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = eventsids[d]
            cts = eventscts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad *                     n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))


# ## Run the model
# 

import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import mongo_client
from bson import ObjectId

project_id = "517eda23c82561f72a000005"

# The number of documents to analyze each k
batchsize = 500
# The total number of documents in Wikipedia
D = mongo_client.get_session_count(project_id)
# The number of topics
K = 6

event_set = mongo_client.get_events_ids_by_project_id(project_id)
W = len(event_set)
model = onlineldavb.OnlineLDA(event_set, K, D)

for k, (n_skip,n_limit) in enumerate(build_batches(n, batch_size)):

    sessions = mongo_client.get_sessions_batch(project_id, n_skip, n_limit)
        
    (gamma, bound) = model.update_lambda(sessions)

    (event_tokens, event_counts) = onlineldavb.parse_sessions_list(sessions, model._event_set)
    pereventsbound = bound * len(sessions) / (D * sum(map(sum, event_counts)))

    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' %         (k, model._rhot, numpy.exp(-pereventsbound))

    if (k % 10 == 0):
        numpy.savetxt('lambda-%d.dat' % k, model._lambda)
        numpy.savetxt('gamma-%d.dat' % k, gamma)





import tensorflow as tf
import numpy as np


# ## simple linear regression with tensorflow
# 

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b


# optimization, minimize mean sq. error
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# need to initialize all of the variables
init = tf.initialize_all_variables()


# create a tensor flow session
sess = tf.Session()
# then launch the graph
sess.run(init)


for step in xrange(201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)


# ## Load the mnist dataset
# 

import os, sys
data_home = os.environ.get('DATA_HOME', None) + "/mnist"
sys.path.insert(0, data_home)

import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


sess.run(tf.initialize_all_variables())


# model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# training function
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# train
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# ## multilayer convolutional network
# 

# for initializing model weights

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# reshape to 4-d tensor, x, image width, image height, color channels
x_image = tf.reshape(x, [-1,28,28,1])


# apply the layer to the input and use the relu activation function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# densly connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# avoid overfitting using 'dropout', a probability that any given neuron's activation is blocked
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# softmax output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# training

# optimization function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# count the num correctly predicted
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# accuracy function
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# unitialize the session
sess.run(tf.initialize_all_variables())

# batch training
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})





# # Find the missing number in n-1 numbers
# 
# You're given an array of unsorted numbers in range 1 to n, of length n-1, with one missing.  Find the missing number
# 
# Solution:
#     
#     The sum of all integers from 1 to n is n*(n+1)/2
#     We can then do this in O(n-1) with constant extra memory
#     
#     Another approach is to create a set/hashtable of numbers from 1 to n, 
#     and walk the array, popping elements from the hash table until we're left with the last one.
#     This os O(2*n) and O(n) extra memory.
# 

from random import shuffle

def find_missing_number(arr):
    n = len(arr) + 1 # O(1)
    expected_sum = int(n * (n + 1) / 2)
    s = sum(arr) # O(n)
    return expected_sum - s

arr = list(range(1,11))
shuffle(arr)
print(arr.pop(4))
print(find_missing_number(arr))


# # Create an LRU cache
# 
# Solution:
#     
#   Use a hashmap and a doubly-linked list with pointers to the head and tail.  The doubly linked list maintains the order, while the hash-map points from the key to the node.  When accessed, pop the element from the doubly-linked list and re-insert it at the front of the list.  To drop an elements from the cache because it times out, pop from the end of the list.
# 

# # Find if a name, from a list of names, exists in a document.  Do this in O(n) time.
# 
# Solution:
# 
# Build a trie/suffix tree from the list of names, with spaces.  This is pre-processing, so it won't effect the run-time cost of the algorithm.  Then walk the documnet, and test the trie at each word.  You can return a full list of names in the doc if you choose.
# 
# This should run in O(k*n), where k is the average length of the words/names in the tree.  For large n, the k is negligable, so basically O(n).
# 

# # Given an array of letters, find the second most frequent item
# 
# 
# Solution:
#     
#     Create a hashmap for the items and their counts.  Also keep two pointers for the first and second most frequent elements.  For each new element, check if it's higher than first, or second highest count, if it is, replace them, and shift down the pointers.  This will take O(n) time and O(n) extra memory.  This is very specific to the kth most frequent element, and becomes very complicated as k becomes larger.
#     
#     There is also a with a max-heap, but this requires O(n*logn), because each insertion takes logn, and you have to insert possibly n times if every item is unique.  We can get the kth most frequent element more easily this way.  However, this is basically as expensive as counting and sorting.
# 

def find_second_most_frequent(arr):
    counts_map = {arr[0] : 1}
    first_most = arr[0]
    idx = 1
    while arr[idx] == first_most:
        idx += 1
        counts_map[arr[idx]] += 1
    
    counts_map[arr[idx]] = 1
    second_most = arr[idx]
    idx += 1
    while arr[idx] == second_most:
        idx += 1
        counts_map[arr[idx]] += 1
    
    if counts_map[second_most] > counts_map[first_most]:
        second_most, first_most = first_most, second_most
    
    n = len(arr)
    # now we walk the rest of the array
    while idx < n:
        i = arr[idx]
        
        # increment
        if i in counts_map:
            counts_map[i] += 1
        else:
            counts_map[i] = 1
        
        # check if it was the second most, and if it's now the first most
        if i == second_most and counts_map[i] > counts_map[first_most]:
            # swap them
            first_most, second_most = second_most, first_most
        
        # check if it's now the second most
        elif not i == first_most and counts_map[i] > counts_map[second_most]:
            # replace the second most
            second_most = i
        idx += 1
    
    if counts_map[second_most] == counts_map[first_most]:
        return (first_most, second_most)
    else:
        return second_most

    
arr1 = ['a','b','c','a','a','a','b','b','b','c','d','d','d','e','e','e','e','e','e','e','e']
print(find_second_most_frequent(arr1))
arr2 = ['a','b','c','a','a','a','b','a','a','a','a','b','b','c','d','d','d','e','e','e','e','e','e','e','e']
print(find_second_most_frequent(arr2))


# # Test if a tree is a valid binary tree
# 
# Solution:
#     
#     DFS or BFS the tree and test if each node has at most two children.
#    
# # Test if a tree is a valid binary search tree
# 
# Solution:
#     use min and max pointers
# 

INT_MAX = 10000000000
INT_MIN =-10000000000
 
# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, val, left = None, right = None):
        self.val = val 
        self.left = left
        self.right = right


def is_BST(root, mi, ma):
    
    def is_BST_rec(node, mi, ma):
        if node is None:
            return True
        
        if node.val < mi or node.val > ma:
            return False
        
        return is_BST_rec(node.left, mi, node.val - 1) and is_BST_rec(node.right, node.val + 1, ma)

    
    return is_BST_rec(root, mi, ma)

root = Node(4,
           Node(2, 
                Node(1), 
                Node(3)
               ),
           Node(5)
           )

print(is_BST(root, INT_MIN, INT_MAX))


# # Get the higth difference of two nodes in a tree
# 
# Solution:
#     
# 
# DFS for both nodes and keep track of depth.  As you find the nodes, update the depth pointers.  Compare at the end.  Run time O(n) to find the nodes with DFS, with constant extra memory, just the two pointers.
# 

class Node:
    
    def __init__(self, val, children = []):
        self.val = val
        self.children = children

def get_depth_difference(root, val1, val2):
    
    depth1 = None
    depth2 = None
    depth = 0
    
    def dfs_rec(node, val1, val2, depth1, depth2, cur_d):
        if not depth1 and not depth2:
            return (depth1, depth2)
        if node.val == val1:
            depth1 = cur_d
        if node.val == val2:
            depth2 = cur_d
        
        for c in node.children:
            dfs_rec(c, val1, val2, depth1, depth2, cur_d+1)
            
    dfs_rec(root, val1, val2, depth1, depth2, 0)
    return abs(depth1 - depth2)


# # Design a media player, to which songs can be added, and can play the songs in random order, without repeats
# 
# Solution:
# 
# We can store the songs in a sequential directory structure, with indices from 0 to n-1. Adding songs requires only appending to sequential directory.  We can play randomly by keeping an array of indices from 0 to n-1 and selecting randomly.  After the first song, we can ensure no repeats by swapping the current item with the item at the end of the index array, and subsiquently sampling from 0 to n-2.
# 

from random import randint

class MediaPlayer:
    
    def __init__(self, songs):
        self.songs = songs
        self.n = len(songs)
        self.indices = list(range(n))
    
    def add_song(self, song):
        self.songs.append(song)
        self.n += 1
        self.indices.append(self.n)
        if n > 1:
            # put newest at second to last to maintain current at index -1
            self.indices[-2], self.indices[-1] = self.indices[-1], self.indices[-2]
    
    def get_song(self, idx):
        # read the song from directory and return it
        return self.song[idx]
    
    def play(self, song):
        # play the song
        print("playing song")
    
    def start(self):
        idx = randint(0,n)
        # put current songs idx at the end
        self.indices[idx], self.indices[-1] = self.indices[-1], self.indices[idx]
        return idx
    
    def next(self):
        idx = randint(0, n-1)
        self.indices[idx], self.indices[-1] = self.indices[-1], self.indices[idx]
        return idx


# # Division without division
# 
# Impliment integer divide method, without using the division operator
# 
# Solutions:
#     
#     Subtraction, and a counter or Bit-shifting
#     
#     
# 

def divide(num, denom, remainder = False):
    c = 0
    
    while num >= denom:
        num -= denom
        c += 1
    
    if remainder:
        return (c, num)
    else:
        return c

print(divide(26, 5))
print(divide(3, 1))
print(divide(310432, 323, True))





# ## Installing Elasticsearch
# 

# If your on mac and can brew install it:
# 
# ` > brew install elasticsearch`
# 
# ` > sudo ln -sfv /usr/local/opt/elasticsearch/*.plist ~/Library/LaunchAgents`
# 
# The default endpoint is `localhost:9200`.  We can test it like so:
# 
# ` > curl localhost:9200`
# 
# `
# {
#   "name" : "Carolyn Trainer",
#   "cluster_name" : "elasticsearch_yourname",
#   "version" : {
#     "number" : "2.0.0",
#     "build_hash" : "de54438d6af8f9340d50c5c786151783ce7d6be5",
#     "build_timestamp" : "2015-10-22T08:09:48Z",
#     "build_snapshot" : false,
#     "lucene_version" : "5.2.1"
#   },
#   "tagline" : "You Know, for Search"
# }
# `
# 
# 
# Here are some other resources:
# 
# [1](https://github.com/sloanahrens/qbox-blog-code/blob/master/ch_1_local_ubuntu_es/install_es.sh) or [2](http://joelabrahamsson.com/elasticsearch-101/)
# 
# Otherwise, google is your friend.
# 

# ## Inserting the Movielens dataset
# 
# A good portion of this proceedure, and some of the code, is taken from [here](https://www.mapr.com/products/mapr-sandbox-hadoop/tutorials/recommender-tutorial)
# 
# [Movielense dataset](http://grouplens.org/datasets/movielens/)
# 
# ` > wget http://files.grouplens.org/datasets/movielens/ml-latest.zip `    
# 
# To create "movielense" mapping:
# 
# `
# curl -XPUT 'http://localhost:9200/movielen' -d '
# {
#   "mappings": {
#     "film" : {
#       "properties" : {
#         "numFields" : { "type" :   "integer" }
#       }
#     }
#   }
# }'
# `
# 
# `{"acknowledged":true}%`
# 
# In `libraray_home/data/movielense` there is a script called "index.py".  This script with convert the movie.csv into a file of paired json docs suitable for inserting into elasticsearch.  The first doc creates the document, and the second specifies the fields.
# 
# Drop `index.py` into the folder with the unzipped movielens data and run:
# 
# ` > python index.py > index.json `
# 
# Then, use curl to bulk insert the data into elasticsearch
# 
# ` > curl -s -XPOST localhost:9200/_bulk --data-binary @index.json; echo`
# 
# There should be a bunch of output, the last element of which should look like:
# 
# ` {"create":{"_index":"movielens","_type":"film","_id":"151711","_version":1,"_shards":{"total":2,"successful":1,"failed":0},"status":201}}]} `
# 

# ## Querying Elasticsearch
# 
# To test out some queries in ES, we can use the chrome plugin Sense (beta).  
# We can search for some dramas with a query like this:
# 
# `
# GET _search
# {
#    "query": {
#       "match": {
#           "genre" : "drama"
#       }
#    },
#    "size" : 8
# }
# `
# 
# We can also curl ES and search like so:
# 
# `
# curl -XPOST "http://localhost:9200/_search" -d'
# {
#     "query": {
#         "query_string": {
#             "query": "kill"
#         }
#     }
# }'
# `
# 
# which should return:
# 
# `
# {"took":14,"timed_out":false,"_shards":{"total":10,"successful":10,"failed":0},"hits":{"total":78,"max_score":3.0908446,"hits":[{"_index":"movielens","_type":"film","_id":"390","_score":3.0908446,"_source":{ "id": "390", "title" : "Faster Pussycat! Kill! Kill!", "year":"1965" , "genre":["Action", "Crime", "Drama"] }},{"_index":"movielens","_type":"film","_id":"94427","_score":3.0209367,"_source":{ "id": "94427", "title" : "Shadow Kill", "year":"2002" , "genre":["Drama"] }},{"_index":"movielens","_type":"film","_id":"132112","_score":2.782512,"_source":{ "id": "132112", "title" : "Good Kill", "year":"2014" , "genre":["Thriller"] }},{"_index":"movielens","_type":"film","_id":"4764","_score":2.6733258,"_source":{ "id": "4764", "title" : "Kill Me Later", "year":"2001" , "genre":["Romance", "Thriller"] }},{"_index":"movielens","_type":"film","_id":"86628","_score":2.6733258,"_source":{ "id": "86628", "title" : "Kill the Irishman", "year":"2011" , "genre":["Action", "Crime"] }},{"_index":"movielens","_type":"film","_id":"132958","_score":2.6733258,"_source":{ "id": "132958", "title" : "The Kill Team", "year":"2013" , "genre":["Documentary", "War"] }},{"_index":"movielens","_type":"film","_id":"61697","_score":2.6226687,"_source":{ "id": "61697", "title" : "Righteous Kill", "year":"2008" , "genre":["Crime", "Mystery", "Thriller"] }},{"_index":"movielens","_type":"film","_id":"70008","_score":2.6226687,"_source":{ "id": "70008", "title" : "Kill Your Darlings", "year":"2006" , "genre":["Comedy", "Drama"] }},{"_index":"movielens","_type":"film","_id":"86677","_score":2.6226687,"_source":{ "id": "86677", "title" : "Kill Theory", "year":"2009" , "genre":["Horror", "Thriller"] }},{"_index":"movielens","_type":"film","_id":"101428","_score":2.6226687,"_source":{ "id": "101428", "title" : "Kill for Me", "year":"2013" , "genre":["Drama", "Thriller"] }}]}}%
# `
# 
# More details about the query DSL can be found [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
# 




# #  Valid parens
# 
# Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
# 
# The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
# 

def has_valid_parens(s):
    """
    :type s: str
    :rtype: bool
    """
    left_to_right = { "{": "}","[" : "]","(" : ")" }
    stack = []
    for i in s:
        if i in left_to_right.keys():
            stack.append(i)
        elif i in left_to_right.values():
            if len(stack) == 0:
                return False
            last = stack.pop()
            if left_to_right[last] == i:
                continue
            else:
                return False
        else:
            continue

    if len(stack) == 0:
        return True
    else:
        return False


print has_valid_parens(")asdf")
print has_valid_parens("]asdf")
print has_valid_parens("}asdf")
print has_valid_parens("{a}(s)[df]")
print has_valid_parens("a[sd(fs)ad]fas{d[fsa]}")


import math
def isPowerOfThree(n):
    """
    :type n: int
    :rtype: bool
    """
    l = n
    while math.sqrt(l) % 1 != 0:
        l = math.sqrt(1)
    
    while n > 27:
        n = n / float(27)
    
    if n == 1 or n == 3 or n == 9 or n == 27:
        return True
    else:
        return False

    
print isPowerOfThree(3)
print isPowerOfThree(9)
print isPowerOfThree(27)
print isPowerOfThree(28)





get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn
from sklearn.datasets import make_blobs

# setup pyspark for IPython_notebooks
spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))


data_home = os.environ.get('DATA_HOME', None)
sys.path.insert(0, data_home)

# data
from gen_data import make_blobs_rdd

# utilitiy functions for this notebook
from lsh_util import *

# make some data
N = 1000
d = 2
k = 5
sigma = 3
bound = 10

data_RDD = make_blobs_rdd(N, d, k, sigma, bound, sc)
data_RDD.take(2)


# # Random projection LSH
# 

def config_random_projection(d, n_hyperplanes = 5, scale = 2.0, seed = None):    
    # random projection vectors
    Z = (np.random.rand(d, n_hyperplanes) - 0.5) * scale
    def projection_func(tup):
        y, x = tup # expect key (int, 1xD vector)
        projs = x.T.dot(Z) # random projections
        bucket = to_bucket(projs)
        return (bucket, y)
    
    return (Z,projection_func)


Z, hash_func = config_random_projection(d)

gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
for b, g, c in sorted(gini_impurities):
    print "bucket: %s , in bucket: %d , gini_impurity: %f" % (b, c, g)


c0 = np.stack(data_RDD.filter(lambda t: t[0] == 0).map(lambda t: t[1]).collect())
c1 = np.stack(data_RDD.filter(lambda t: t[0] == 1).map(lambda t: t[1]).collect())
c2 = np.stack(data_RDD.filter(lambda t: t[0] == 2).map(lambda t: t[1]).collect())
c3 = np.stack(data_RDD.filter(lambda t: t[0] == 3).map(lambda t: t[1]).collect())
c4 = np.stack(data_RDD.filter(lambda t: t[0] == 4).map(lambda t: t[1]).collect())

plt.scatter(c0[:,0],c0[:,1],color='g')
plt.scatter(c1[:,0],c1[:,1],color='y')
plt.scatter(c2[:,0],c2[:,1],color='b')
plt.scatter(c3[:,0],c3[:,1],color='k')
plt.scatter(c4[:,0],c4[:,1],color='m')

# projection vectors
plt.scatter(Z.T[:,0],Z.T[:,1],color='r',s=50)


# impurity as we scale up the number of hyperplanes used for projections

for n_Z in range(10,201,10):
    Z, hash_func = config_random_projection(d, n_Z)
    gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
    g_i = weighted_gini(gini_impurities)
    print "%d projections, gini_impurity: %f" % (n_Z, g_i)
    





# # Sparse Matrix
# 
# Create a sparse matrix class, impliment the following methods: 
# - set(col, row, val) # (int, int, float/double)
# - sum(col, row) # sum the sub-matrix from (0,0) to (col, row)
# 

class SparseMatrix:
    
    def __init__(self, nCols, nRows, elements):
        """ Initialize with a size, and some elements. """
        self.num_cols = num_cols
        self.num_cows = num_rows
        self.rows = {}
        for e in elements:
            self.set(e.col, e.row, e.val)
        
    
    def set(self, col, row, val):
        """ update a value inplace, indexes start at zero """
        if col < 0 or col > (self.num_cols - 1):
            raise Exception('col out of bounds')
        elif row < 0 or row > (self.num_rows - 1):
            raise Exception('row out of bounds')
        elif row in self.rows:
            self.rows[row][col] = val
        else:
            self.rows[row] = {col: val}
    
    def sum(self, col, row):
        """ Sum the sub-matrix from (0,0) to (row, col)"""
        """ this can be faster with better data-structures """
        
        for r, cols in self.rows.items():
            s = 0
            if r <= row:
                for c, v in cols.items():
                    if c <= col:
                        s += v
        return s


# # Convert a number to english
# 
# def convert(100) => "One Hundred"
# 

def convert(number):
    
    firstTwenty = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine","ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "twenty", "thirty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninty"]
    thousands = ["", "thousand", "million", "billion", "trillion"]
    
    def teen_string(num):
        if num < 20:
            return firstTwenty[num]
        else:
            return tens[num/20] + " " + firstTwenty[num%20]
    
    def hundreds_string(num):
        output = ""
        if (num / 100) > 0:
            output += firstTwenty[num / 100] + " and "
        output += teen_string(num % 100)
        return output
    
    if number < 1000:
        output = hundreds_string(number)
    else:
        power = 1
        output = hundreds_string(number)
        while number > 0:
            pNum = number % 1000
            output = hundreds_string(pNum) + " " + thousands[power]  + ", " + output
            power += 1
            number = number / 1000
    
    return output


# # Flatten
# 
# Given a graph of nodes, where the node type is like so, Node(val, down, right), return a flattened linked list, where we traverse down before right. 
# 
# Nodes pointed to with a down pointer do not have right pointers.
# 

import collections

Node = collections.namedtuple('Node', ['val', 'down', 'right'])

def flatten(head):
    tail = head
    
    def flattenRec(root, output = None):
        tail.right = root
        tail = root
        while tail.down:
            tail.right = tail.down
            tail = tail.down
        if root.right:
            return flattenRec(root.right)
        else:
            return head
        
    return flattenRec(head)


# # Print Tree columns
# 
# Given a binary tree, print the nodes in order of the column they are in, from left to right.  Within the column, print from top to bottom.
# 
# <img src="print_tree_columns.png">
# 
# 

import collections

TreeNode = collections.namedtuple('TreeNode', ['val', 'left', 'right'])

tree_root = TreeNode(6, 
            TreeNode(3,
                 TreeNode(5,
                      TreeNode(9, None, None),
                      TreeNode(2, 
                           None, 
                           TreeNode(7, None, None)
                          )
                     ),
                 TreeNode(1,None,None)
                ),
            TreeNode(4,
                 None,
                 TreeNode(0,
                     TreeNode(8, None, None), 
                     None
                     )
                )
           )


"""
Solution:


Also create an auxilary hashmap of column_number -> array[int], 
    where the ints will be the TreeNode values, in order from top to bottom.
    
We then do a pre-order, depth first traversal, where we keep track of the column number by starting at 0 at the root, 
    then modifying the column number -= 1 when we go left, and += 1 when we go right.
    For each node, we check if the column number is in the map, if it is, 
        we append the current nodes value to the list at that column number, if it is not, 
        we add a new column number key, and a new list with that new node value

After the traversal, we get the keys from the map, and sort them in ascending order.  
    We walk the ordered keys and print the node values at that key, in the order they were inserted.

run time will be O(n) most of the time, unless the tree is degenerate and has a lot of columns.
    In which case, the run-time could be as bad as O(nlogn) because of the sorting of the column numbers
"""
        
        
def print_tree_columns(root):
    columns = {}
    def preorder_dfs_traversal(node, col_num):
        if node is not None:
            if col_num in columns:
                columns[col_num].append(node.val)
            else:
                columns[col_num] = [node.val]
            preorder_dfs_traversal(node.left, col_num - 1)
            preorder_dfs_traversal(node.right, col_num + 1)
    preorder_dfs_traversal(root, 0)
    
    output = ""
    for col_num in sorted(columns.keys()):
        for val in columns[col_num]:
            output += str(val) + " "
    print(output)





# # Reservoir sampling
# 
# Is a sampling method use to collect iid samples from am unbounded stream of data.
# 
# The concept is that you have a stream of valus comming in starting at index 0, and growing to index n, where n is increasing forever.  You have a sample from this stream of size k, where each value has an equal probability of being in the sample (k/n).
# 
# The algorithm:
#     For the first k values, collect them all and put them in your sample array.
#     For the n = k+1, or the k+1th value in the stream, 
#         keep the new item with probability k/n,
#         if we choose to keep the new item, we replace one of the current samples with the new sample, with a uniform probability (1/k),
#         This means that the probability of each item remaining in the sample is (k/n) * (1/k), or (1/n), which is what we want.
# 

from random import randint

class ReservoirSampler:
    
    def __init__(self, sample_size):
        self.samples = []
        self.sample_size = sample_size
        self.seen_values = 0
    
    def is_full(self):
        return len(self.samples) == self.sample_size
    
    def update_sample(self, new_value):
        self.seen_values += 1
        if not self.is_full():
            # collect the first k
            self.samples.append(new_value)
        else:
            # select value, j, between (0,n-1)
            # it has a k/n probability of being in range (0,k-1)
            # if it is, replace samples[j] with the new value
            j = randint(0,self.seen_values)
            if j < self.sample_size:
                self.samples[j] = new_value
        

r_sampler = ReservoirSampler(25)
for i in range(1000):
    r_sampler.update_sample(i)

print(r_sampler.samples)





get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn
from sklearn.datasets import make_blobs

# setup pyspark for IPython_notebooks
spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))


import nltk
from nltk.book import *


# ## nltk niceties
# 
# the 'nltk.text.Text' class
# 
# concordance:  finding a word with some context from a document
# 

text1.concordance("monstrous")


# similar words based on context
# 

text1.similar("monstrous")


# .common_context : of two or more words
# 

text2.common_contexts(["monstrous", "very"])


# .dispersion: the location of words in the text
# 

# text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


# Simple metrics:
# 
# diversity
# 

def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total


# ## Accessing data from the web
# 

from BeautifulSoup import BeautifulSoup          # For processing HTML
#from BeautifulSoup import BeautifulStoneSoup     # For processing XML
#import BeautifulSoup                             # To get everything
from urllib import *


url = "http://www.gutenberg.org/files/2554/2554.txt"
raw = urlopen(url).read()
print type(raw)
print len(raw)
print raw[:75]


# ## tokenizing with library tokenizer
# 

tokens = nltk.word_tokenize(raw)
print type(tokens)
print len(tokens)
print tokens[:10]


# ## convert to nltk.text.Text
# 

text = nltk.Text(tokens)
print type(text)
print text[1020:1060]
print text.collocations()


# ## Getting HTML and scraping
# 

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read()
print html[:60]
soup = BeautifulSoup(html) # BeautifulSoup class
raw = soup.getText()
tokens = nltk.word_tokenize(raw)
print tokens[96:399]





get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn

PROJECT_HOME = os.environ.get('PROJECT_HOME', None)
sys.path.insert(0, PROJECT_HOME + "/util")
from loaders import get_english_dictionary


class Trie_dict:
    
    def __init__(self):
        self._end = '_end_'
        self._root = dict()
    
    def insert(self, word):
        current_dict = self._root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict[self._end] = self._end
    
    def insert_batch(self, words):
        for word in words:
            self.insert(word)
    
    def view(self):
        print(self._root)
    
    def view_root_keys(self):
        print(self._root.keys())
    
    def contains(self, word):
        current_dict = self._root
        for letter in word:
            if letter in current_dict:
                current_dict = current_dict[letter]
            else:
                return False
        # the _end flag indicates this is the end of a word
        # if it's not there, the word continues
        if self._end in current_dict:
            return True
        else:
            return False
    

    def suggest(self, partial, limit = 5):
        """
        Since this trie doesn't store frequency of words as it trains, we're just 
        going to return the alphabetically first 'limit', shortest, terms.
        """
        suggestions = []

        def suggest_dfs(partial_dict, partial ):
                if len(suggestions) < limit:
                    for ch in sorted(partial_dict.keys()): 
                        # sorting by alpha, this happens to give us _end_ first
                        # could be pre-sorting by frequency for better 
                        #   speed and smarted recommendations
                        if len(suggestions) >= limit:
                            break
                        elif ch == self._end:
                            suggestions.append(partial)
                        else:
                            # recurse
                            suggest_dfs(partial_dict[ch], partial + ch)

        partial_dict = self._find_patial(partial)
        if not partial_dict == None:
            suggest_dfs(partial_dict, partial)
        
        return suggestions

    def _find_patial(self, partial):
        top_dict = self._root
        for char in partial:
            if char in top_dict:
                top_dict = top_dict[char]
            else:
                # there are no words starting with this sequence
                return None
        return top_dict

        


# A note on the dictionary.set_default(key, default_val) method.  
# This method is equivilant to a method that looks like this:
def set_default(dictionary, key, default_val = {}):
    if key in dictionary:
        return dictionary[key]
    else:
        dictionary[key] = default_val
        return dictionary[key]


trie = Trie_dict()
trie.insert_batch(get_english_dictionary())


print("Suggestions")
print("")
print("'reac': ")
print(trie.suggest("reac"))
print( "")
print( "'poo': ")
print( trie.suggest("poo"))
print( "")
print( "'whal': ")
print( trie.suggest("whal"))
print( "")
print( "'dan': ")
print( trie.suggest("dan"))
print( "")


# # Tries with some statistical flavor
# 
# ## Trie with frequency distribution
# Create a Trie where we keep track of how many times we've gone down each branch of the tree.  We can use this distribution over suggestions to rank our suggestions.
# 
# This prob. can be expressed as P( next_word = word_i | incomplete)
# 

class Trie_Statistical:
    
    def __init__(self):
        self._end = '_end_'
        self._root = dict()
        self._total_words = 0
        self._search_limit = 100
    
    def insert(self, word):
        current_dict = self._root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        # keep counts at { last_letter : {'_end_' : count} }
        if self._end in current_dict:
            current_dict[self._end] += 1
        else:
            current_dict[self._end] = 1
        self._total_words += 1
        
    
    def insert_batch(self, words):
        for word in words:
            self.insert(word)
    
    def view(self):
        print(self._root)
    
    def view_root_keys(self):
        print(self._root.keys())
    
    def _normalize_suggestion_probs(self, suggestions):
        total = 0
        for w, c in suggestions:
            total += c
        for i, t in enumerate(suggestions):
            suggestions[i] = (t[0], t[1] / total)
    
    def contains(self, word):
        current_dict = self._root
        for letter in word:
            if letter in current_dict:
                current_dict = current_dict[letter]
            else:
                return False
        # the _end flag indicates this is the end of a word
        # if it's not there, the word continues
        if self._end in current_dict:
            return True
        else:
            return False
    
    def suggest(self, partial, limit = 5):
        """
        """
        suggestions = []

        def suggest_dfs(partial_dict, partial ):
                if len(suggestions) < self._search_limit:
                    for ch in sorted(partial_dict.keys()): 
                        # sorting by alpha, this happens to give us _end_ first
                        # could be pre-sorting by frequency for better 
                        #   speed and smarter recommendations
                        if len(suggestions) >= self._search_limit:
                            break
                        elif ch == self._end:
                            suggestions.append((partial, partial_dict[self._end]))
                        else:
                            # recurse
                            suggest_dfs(partial_dict[ch], partial + ch)

        partial_dict = self._find_patial(partial)
        if not partial_dict == None:
            suggest_dfs(partial_dict, partial)
        
        self._normalize_suggestion_probs(suggestions)
        sorted_suggestions = sorted(suggestions, key=lambda pair: pair[1])
        if limit > 0:
            return sorted_suggestions[:limit]
        else:
            return sorted_suggestions


    def _find_patial(self, partial):
        top_dict = self._root
        for char in partial:
            if char in top_dict:
                top_dict = top_dict[char]
            else:
                # there are no words starting with this sequence
                return None
        return top_dict


trie = Trie_Statistical()
# we're reading a dictionary, so we will have 1 example of every word.
trie.insert_batch(get_english_dictionary())


print("Suggestions")
print("")
print("'reac': ")
print(trie.suggest("reac"))
print( "")
print( "'poo': ")
print( trie.suggest("poo"))
print( "")
print( "'whal': ")
print( trie.suggest("whal"))
print( "")
print( "'dan': ")
print( trie.suggest("dan"))
print( "")


# # More advanced Tries
# 
# ## Trie with simple Markov-Transition Distribution
# We can use some sentance context to make suggestions as well.  We can build a transition matrix from work X to work Y (represented sparsely because the # of words is likely huge), to get the probability of the the next word, given the last word, or 
# 
# P( next_word = word_i | incomplete, last_word = word_j) = P( next_word = word_i | incomplete) * P( next_word = word_i | last_word = word_j )
# 
# ## HMMs
# We can extend the Markov toolkit even further, by modeling the word sequence as a Hidden-Markov Model.  The Hidden-Markov model creates a tractible way of computing not just P( next_word = word_i | last_word = word_j ) but P( next_word = word_i | last_word = word_j, last_last_word = word_j, ..., all the way to firs_word = word_x ).
# 
# HMMs are a whole different beast, but once you've got one, you can update your ranking of the next word with the following:
# 
# P( next_word = word_i | incomplete, all_previous_words) = P( next_word = word_i | incomplete) * P( next_word = word_i | all_previous_words )
# 










# ## Find shortest distance point
# 
# On a given 2-D grid, find the point that is shortest distance from any empty point (0) of the objectives (1), not passing through a blocked point (2).
# 
# 
# For example, given three objectives at (0,0), (0,4), (2,2), and an obstacle at (0,2):
# 
# 
# ```
# 1 - 0 - 2 - 0 - 1
# |   |   |   |   |
# 0 - 0 - 0 - 0 - 0
# |   |   |   |   |
# 0 - 0 - 1 - 0 - 0
# ```
# 
# would be represented as a list of lists, like so:
# 
# `[[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]`
# 

def shortestDistance(self, grid):
    if not grid or not grid[0]: 
        return -1
    
    rows, cols = len(grid), len(grid[0])
    objectives = sum(val for line in grid for val in line if val == 1)
    
    hit = [[0] * cols for i in range(rows)]
    distSum = [[0] * cols for i in range(rows)]

    def BFS(start_x, start_y):
        visited = [[False] * cols for k in range(rows)]
        visited[start_x][start_y] = True
        
        count1 = 1
        queue = collections.deque([(start_x, start_y, 0)])
        
        while queue:
            x, y, dist = queue.popleft()
            for i, j in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= i < rows and 0 <= j < cols and not visited[i][j]:
                    visited[i][j] = True
                    if not grid[i][j]:
                        queue.append((i, j, dist + 1))
                        hit[i][j] += 1
                        distSum[i][j] += dist + 1
                    elif grid[i][j] == 1:
                        count1 += 1
        return count1 == objectives  

    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1:
                if not BFS(x, y): return -1
    
    return min([distSum[i][j] for i in range(rows) for j in range(cols) if not grid[i][j] and hit[i][j] == objectives] or [-1])





# # Remove numbers in an array
# 
# Given an array, and a number, remove all instances of that number from the array, and return the edited array and it's new length.
# 
# Solution:
# For a mutable array, we can perform this operation in O(n), with one pass, with O(1) extra memory usage.
# We can also do this while maintaining the original order of the items.
# 
# 
# The process involves removing and re-inserting every item as we walk the array of length (l).  While doing so, we keep a counter of the number of times we've seen the item to delete (k).  At every index (i), we check the item at that index is the item to delete.  If it is, we incriment our counter, and move on.  If not, we copy that item to index (i-k).  At the end, we drop the last K items by return the portion of the array from 0 until l - k.
# 

def remove_item_from_array(arr, item):
    k = 0
    l = len(arr)
    for idx, j in enumerate(arr):
        if j == item:
            k += 1
        else:
            arr[idx - k] = j
    
    return arr[0:(l-k)]


arr1 = [0,1,2,3,0,4,5,6,7,0,8,0,9]
arr2 = [9,2,3,4,1,5,5,6,0,0,0,3,0]
arr3 = [0,0,0,0,0,0,0,1,0]

print(remove_item_from_array(arr1,0))
print(remove_item_from_array(arr2,0))
print(remove_item_from_array(arr3,0))
print(remove_item_from_array(arr3,1))


# # String path in a matrix of chars
# 
# Given a matrix of Chars, and a desired string, find if the string is in the matrix; the chars must be continuously connected but can go in any direction (up, down, left or right) and can change direction mid-string.
# 
# 
# Solution:
# 
# Walk the matrix, in any order, but whenever we get the the first char of the string, begin a DFS in the four directions.
# 

def test_cell(y_i, x_i, s, mat):
    return (y_i > 0) and (x_i > 0) and (y_i < len(max)) and (x_i < len(mat[0])) and (s[0] == mat[y_i][x_i])

def strDFS(y, x, s, mat):
    if len(s) == 0:
        return True
    # up
    elif test_cell(y-1, x, s, mat):
        strDFS(y-1, x, s[1:], mat)
    # down
    elif test_cell(y+1, x, s, mat):
        strDFS(y+1, x, s[1:], mat)
    # left
    elif test_cell(y, x-1, s, mat):
        strDFS(y, x-1, s[1:], mat)
    # right
    elif test_cell(y, x+1, s, mat):
        strDFS(y, x+1, s[1:], mat)
    else:
        return False

def find_string_in_char_matrix(mat, s):
    found = []
    if not ((len(s) == 0) or (len(mat) == 0)):
        for i in range(len(mat[0])):
            for j in range(len(mat)):
                if (mat[j][i] == s[0]) and strDFS(j, i, s[1:], mat):
                    found.append((j,i))
    return found


# # Search a rotated, sorted array
# 
# An array is rotated if some elements from the front are moved to the end, maintaining their order.  Given an array that was sorted in ascending order, with some rotation, perform a search for an item.  Assume there are no duplicates.
# 

def search_in_rotation(arr, lP, rP, item):
    if lP > rP:
        return None
    elif item == arr[lP]:
        return lP
    elif item == arr[rP]:
        return rP
    
    mP = lP + ((rP - lP) // 2)
    if arr[mP] == item:
        return mP
    
    if arr[lP] < arr[mP]:
        # in first increasing array
        if item > arr[lP] and item < arr[mP]:
            # search left
            return search_in_rotation(arr, lP+1, mP-1, item)
        else:
            # search right
            return search_in_rotation(arr, mP+1, rP-1, item)
    else:
        # in the second increasing array
        if item > arr[mP] and item < arr[rP]:
            # search right
            return search_in_rotation(arr, mP+1, rP-1, item)
        else:
            # search left
            return search_in_rotation(arr, lP+1, mP-1, item)

def binary_search_rotated_array(array, item):
    l = len(array)
    if l == 0:
        return None
    elif l == 1 and array[0] == item:
        return 0
    
    return search_in_rotation(array, 0, len(array)-1, item)
        
    
a = list(range(20))
a = a[5:] + a[:5]
print(a)
print(binary_search_rotated_array(a, 15))


# # Min splits to make palandromes from a string
# 
# Strings can be broken into n sub-strings of length >= 1 which are all palandromes. 
# 

def is_panadrome(s):
    if len(s) == 1 or len(s) == 2:
        return True
    else:
        m = len(s) // 2
        # O(n)
        return s[:m] == s[-m:][::-1]

def min_palandrome_splits(string):
    if len(string) == 0:
        return -1
    elif is_panadrome(string):
        return 0
    else:
        # the last element contains the minimum, worst case min_splits == n
        splits = list(range(1,len(string)+1))
        
        for i in range(len(string)):
            if is_panadrome(string[:i]):
                splits[i] = 0
            
            for j in range(0,i):
                if is_panadrome(string[j+1:i]) and splits[i] > splits[j] + 1:
                    splits[i] == splits[j] + 1
        
    return splits[-1]

s = "madamifmadam"
print(is_panadrome(s))
print(min_palandrome_splits(s))

    


# # Given two binary trees, return the first pair of leaves that are non-matching
# 
# ex:
# 
#     tree1: A, B, C, D, E, None, None
#     tree2: A, D, B
#     
#     return: (E,B)
#    
# Trivial solution: Do a DFS on both threes and store the leaves in two arrays.  After completing the DFSs, compare the arrays and return the first non-matching leaves.  Run-time O(2* (N + M) ), with O(N+M) extra memory, since we don't know of these are balanced binary trees.
# 

class TreeNode:
    
    def __init__(self, key, left=None, right=None):
        self.left = left
        self.right = right
        self.key = key
    
    def is_leaf(self):
        if self.left or self.right:
            return False
        else:
            return True


tree1 = TreeNode('A', 
                TreeNode('B',
                         TreeNode('D'),
                         TreeNode('E')
                        ),
                TreeNode('C')
                )


tree2 = TreeNode('A', TreeNode('D'), TreeNode('B'))


leaves = [[],[]]

def collectLeaves(tree1, tree2):
    
    # pre-order dfs
    def dfs(node, tree_num):
        if node.is_leaf():
            leaves[tree_num].append(node.key)
        if node.left:
            dfs(node.left, tree_num)
        else:
            leaves[tree_num].append(None)
        if node.right:
            dfs(node.right, tree_num)
        else:
            leaves[tree_num].append(None)
    
    dfs(tree1, 0)
    dfs(tree2, 1)

    

def compare_leaves(leaves):
    if len(leaves[0]) > len(leaves[1]):
        for i in range(len(leaves[0])):
            l1 = leaves[0][i]
            if i > len(leaves[1])-1:
                l2 = None
            else:
                l2 = leaves[1][i]
            if not l1 == l2:
                return (l1,l2)
    else:
        for i in range(len(leaves[1])):
            l2 = leaves[1][i]
            if i > len(leaves[0])-1:
                l1 = None
            else:
                l1 = leaves[0][i]
            if not l1 == l2:
                return (l1,l2)
    
    return (None, None)
# there is a weird bug in traversing the second tree, it goes right then left
collectLeaves(tree1, tree2)
print(leaves)
compare_leaves(leaves)


# # Sort an array of integers squared
# 
# Given an array of integers, either positive or negative, sort the square of those integers.
# 

def sort_squared(arr):
    negs = []
    output = []
    for i in arr:
        j = i**2
        if i < 0:
            negs.append(j)
        else:
            if len(negs) > 0 and j >= negs[-1]:
                output.append(negs.pop())
            output.append(j)
    # if there are negatives left
    
    for j in negs[::-1]:
        output.append(j)
    
    return output


a = list(range(-10,10))
sort_squared(a)


# # Compute processing time while awaiting repeated tasks
# 
# Given a list of tasks to run, [A,B,C,D,A,F,C] where each task takes 1 unit, except there a wait-time (k) to run a repeated task, if k = 3 [A,B,C,wait1 A, wait2 C, wait3 C], return the run time of the list of tasks.
# 
# 
# Solution:
# 
# Use an auxilliary hashmap, 'task -> last time completed', {taskId : last time completed }, and a total time spent.  Walk through the list, check if the task is in the map.  If so, check if the (current time - last time < k).  If so, jump forward in time to (last time + k).  Incriment the timer by one for each iteration, and put (taskId -> current time) into the map.
# 
# This will take O(n) to run, since you walk through the map once. You'll need O(n) memory, because you need to use an auxilliary hashmap, which could have up to n unique tasks in it.
# 

def compute_processing_time(tasks, k):
    if k == 0:
        return len(tasks)
    if len(tasks) == 1:
        return 1
    
    time = 0
    last_time = {}
    
    for task in tasks:
        if task in last_time and (time - last_time[task] < k):
            time = last_time[task] + k
        time += 1
        last_time[task] = time
    
    return time

tasks1 = ['A','B','C','D'] # k = 3, 4
tasks2 = ['A','B','A','C'] # k = 3, 6
tasks3 = ['A','A','A','A'] # k = 4, 16

print(compute_processing_time(tasks1, 3))
print(compute_processing_time(tasks2, 3))
print(compute_processing_time(tasks3, 4))


# # Process files, compute avg and totals
# 
# 
# You have a folder full of .bin files, that are proprietary, and a class BinToTsvConverter with a method .convert(filename), which converts the .bin file to a .tsv file.  The .tsv file has the schema, and no header:
# 
#     Total_Connections Latency Bandwidth
#     65                  70     20
# 
# Calculate the average latency and total bandwidth.
# 
# This should take us O(N) where N is the total number of rows in all of the files combined.  If we've got a huge amount of files, we can read the files in parallel and compute the statistics per file, then combine the statistics (MapReduce type stuff).
# 

class BinToTsvConverter:

    def __init__(self):
        """
        initializes the class
        """
    
    def convert(self,filename):
        """
        converts the file and writes filename.tsv
        """

def read_calculate_latency_bandwidth(filenames):
    
    connections_total = 0
    latency_total = 0
    bandwidth_total = 0
    rows_total = 0
    
    converter = BinToTsvConverter()
    
    for filename in filenames:
        converter.convert(filename)
        with open("{}.tsv".format(filename)) as f:
            for line in f.readlines():
                connections, latency, bandwidth = line.split("\t")
                connections_total += connections
                latency_total += latency
                bandwidth_total += bandwidth
                rows_total += 1
    
    return (latency_total / rows_total, bandwidth_total)


# map-reduce style

def read_file_calc_stats(filename):
    converter = BinToTsvConverter()
    converter.convert(filename)
    
    connections_total = 0
    latency_total = 0
    bandwidth_total = 0
    rows_total = 0
    
    with open("{}.tsv".format(filename)) as f:
        for line in f.readlines():
            connections, latency, bandwidth = line.split("\t")
            connections_total += connections
            latency_total += latency
            bandwidth_total += bandwidth
            rows_total += 1
    
    return (connections_total, latency_total, bandwidth_total, rows_total)

def combine_stats(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2], t1[3] + t2[3])

def compute_latency_bandwidth(summary_stats):
    return (summary_stats[1] / summary_stats[3], summary_stats[2])

# compute_latency_bandwidth(reduce(combine_stats, map(read_file_calc_stats, filenames)))


# # Find 3 numbers that sum to zero in an array
# 
# Given an array of ints, and a int (3), find all combinations of 3 of the numbers, without repeating any numbers, that sum to zero.
# 
# Solution:
# 
# The trick here is to first convert the items to a set/hashtable, to remove duplicates.  This set also allows us to do fast lookups for items.
# We then need to do a two item deep DFS through the set, where we for each item A, one-by-one grab every other item B, then do set.contains(-(A + B)).  If that item is in there, we add the triplet to the output, otherwise we move on to the next pair.
# 
# To get an output without duplicates we need to do some more trickery.  The output should also be a set/hashtable.  We need to 
# 
#     1) put the triple of values into an array
#     2) sort the array (trivial since it's three items)
#     3) create a comma-separated string from that array 
#     4) then upsert the string into the set/hashtable.
# 
# It's up to you if you want to post-process the strings and trurn numbers.
# 
# Run-time should be O(n * (n-1) / 2), since we don't have to compare two items twice e.g. (A,B) and (B,A).  We'll need O(2*n) extra space for the set/hashtable and an iterable-subscriptbable, and O(3*n) for the output array.  In practice, those memory usage numbers should be much lower.
# 
# This problem can be expanded to arbitrary #s of items and any base value, not just zero.
# 

def find_zero_triples(arr):
    s = set(arr)
    l = list(s)
    output = set()
    
    for i, a in enumerate(l[:-2]):
        for b in l[i+1:]:
            c = -(a+b)
            if not (c == a) and not (c == b) and c in s:
                # these aren't unique solutions
                output.add(",".join(sorted([a,b,c])))
                
    return output

a = list(range(-10,10))
find_zero_triples(a)





