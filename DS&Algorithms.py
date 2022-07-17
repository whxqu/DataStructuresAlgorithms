class Node:
	def __init__(self, value, next_node = None):
		self.value = value
		self.next_node = next_node
	def get_next_node(self):
		return self.next_node
	def get_value(self):
		return self.value
	def set_next_node(self, next_node):
		self.next_node = next_node

class Linked_List:
	def __init__(self, value = None):
		self.head_node = Node(value)
	def push(self, value):
		new_head = Node(value)
		new_head.set_next_node(self.head_node)
		self.head_node = new_head
	def remove(self, value):
		current_node = self.head_node
		if current_node.get_value() == value:
			self.head_node.set_next_node(current_node)
		else:
			while current_node:
				next_node = current_node.get_next_node()
				if next_node.get_value() == value:
					current_node.set_next_node(next_node.get_next_node())
					break
				current_node = current_node.get_next_node()
	def listify(self):
		lst = []
		current_node = self.head_node
		while current_node:
			if current_node.get_value() != None:
				lst.append(current_node.get_value())
			current_node = current_node.get_next_node()
		return lst
	def reverse(self):
		prev_node = None
		current_node = self.head_node
		next_node = current_node.get_next_node()
		while current_node:
			current_node.set_next_node(prev_node)
			prev_node = current_node
			current_node = next_node
			if next_node:
				next_node = next_node.get_next_node()
		self.head_node = prev_node
	def get_nth_from_last(self, n):
		tail_seeker = self.head_node
		current_node = None
		count = 0
		while tail_seeker:
			count += 1
			tail_seeker = tail_seeker.get_next_node()
			if count >= n + 1:
				if current_node is None:
					current_node = self.head_node
				else:
					current_node = current_node.get_next_node()
		return current_node.get_value()
	def swap(self, val1, val2):
		if val1 == val2:
			print("These are the same")
		node1 = self.head_node
		node2 = self.head_node
		node1_prev = None
		node2_prev = None
		while node1:
			if node1.get_value() == val1:
				break
			node1_prev = node1
			node1 = node1.get_next_node()
		while node2:
			if node2.get_value() == val2:
				break
			node2_prev = node2
			node2 = node2.get_next_node()
		if node1 is None or node2 is None:
			print("One or both of these do not exist")
		if node1_prev is None:
			self.head_node = node2
		else:
			node1_prev.set_next_node(node2)
		if node2_prev is None:
			self.head_node = node1
		else:
			node2_prev.set_next_node(node1)
			self.head_node = node2
		temp = node1.get_next_node()
		node1.set_next_node(node2.get_next_node())
		node2.set_next_node(temp)

class Double_Node:
	def __init__(self, value, next_node = None, prev_node = None):
		self.value = value
		self.next_node = next_node
		self.prev_node = prev_node
	def get_next_node(self):
		return self.next_node
	def get_prev_node(self):
		return self.prev_node
	def get_value(self):
		return self.value
	def set_next_node(self, next_node):
		self.next_node = next_node
	def set_prev_node(self, prev_node):
		self.prev_node = prev_node

class Doubly_Linked_List:
	def __init__(self):
		self.head_node = None
		self.tail_node = None
	def add_to_head(self, value):
		new_head = Double_Node(value)
		current_node = self.head_node
		if current_node != None:
			new_head.set_next_node(current_node)
			current_node.set_prev_node(new_head)
		self.head_node = new_head
		if self.tail_node is None:
			self.tail_node = new_head
	def add_to_tail(self, value):
		new_tail = Double_Node(value)
		current_node = self.tail_node
		if current_node != None:
			current_node.set_next_node(new_tail)
			new_tail.set_prev_node(current_node)
		self.tail_node = new_tail
		if self.head_node is None:
			self.head_node = new_tail
	def remove_from_head(self):
		removed_head = self.head_node
		if removed_head is None:
			return None
		self.head_node = removed_head.get_next_node()
		if self.head_node != None:
			self.head_node.set_prev_node(None)
		if self.tail_node == removed_head:
			self.remove_from_tail()
	def remove_from_tail(self):
		removed_tail = self.tail_node
		if removed_tail is None:
			return None
		self.tail_node = removed_tail.get_prev_node()
		if self.tail_node != None:
			self.tail_node.set_next_node(None)
		if self.head_node == removed_tail:
			self.remove_from_head()
	def remove_by_value(self, value):
		current_node = self.head_node
		node_to_remove = None
		while current_node:
			if current_node.get_value() == value:
				node_to_remove = current_node
				break
			current_node = current_node.get_next_node()
		if node_to_remove is None:
			print("This does not exist")
		if node_to_remove == self.head_node:
			self.remove_from_head()
		if node_to_remove == self.tail_node:
			self.remove_from_tail()
		prev_node = node_to_remove.get_prev_node()
		next_node = node_to_remove.get_next_node()
		prev_node.set_next_node(next_node)
		next_node.set_prev_node(prev_node)
	def add_after(self, value1, value2):
		new_node = Double_Node(value2)
		current_node = self.head_node
		if current_node is None:
			return None
		else:
			while current_node:
				next_node = current_node.get_next_node()
				if current_node.get_value() == value1:
					current_node.set_next_node(new_node)
					new_node.set_prev_node(current_node)
					new_node.set_next_node(next_node)
					next_node.set_prev_node(new_node)
				current_node = current_node.get_next_node()
	def display(self):
		lst = []
		current_node = self.head_node
		while current_node:
			if current_node.get_value() != None:
				lst.append(current_node.get_value())
			current_node = current_node.get_next_node()
		return lst

class Queue:
	def __init__(self, max_size = None):
		self.max_size = max_size
		self.head = None
		self.tail = None
		self.size = 0
	def is_empty(self):
		return self.size == 0
	def has_space(self):
		if self.max_size is None:
			return True
		else:
			return self.max_size > self.get_size()
	def get_size(self):
		return self.size
	def peek(self):
		if not self.is_empty():
			return self.head.get_value()
		else:
			print("This queue is empty!")
	def enqueue(self, value):
		if self.has_space():
			new_item = Node(value)
			if self.is_empty():
				self.head = new_item
				self.tail = new_item
			else:
				self.tail.set_next_node(new_item)
				self.tail = new_item
			self.size += 1
		else:
			print("This queue has no more room!")
	def dequeue(self):
		if not self.is_empty():
			removed_item = self.head
			if self.get_size() == 1:
				self.head = None
				self.tail = None
			else:
				self.head = removed_item.get_next_node()
			self.size -= 1
		else:
			print("This queue is empty!")

class Stack:
	def __init__(self, limit = None):
		self.limit = limit
		self.size = 0
		self.top_item = None
	def is_empty(self):
		return self.size == 0
	def has_space(self):
		if self.limit is None:
			return True
		else:
			return self.limit > self.get_size()
	def get_size(self):
		return self.size
	def peek(self):
		if not self.is_empty():
			return self.top_item.get_value()
		else:
			print("This stack is empty!")
	def push(self, value):
		if self.has_space():
			new_item = Node(value)
			if self.is_empty():
				self.top_item = new_item
			else:
				new_item.set_next_node(self.top_item)
				self.top_item = new_item
			self.size += 1
		else:
			print("This stack is no more room!")
	def pop(self):
		if not self.is_empty():
			removed_item = self.top_item
			if self.get_size() == 1:
				self.top_item = None
			else:
				self.top_item = removed_item.get_next_node()
			self.size -= 1
		else:
			print("This stack is empty!!")

class HashMap:
	def __init__(self, array_size):
		self.array_size = array_size
		self.array = [None for items in range(self.array_size)]
	def hash(self, key, count_collisions = 0):
		key_byte = key.encode()
		hash_code = sum(key_byte)
		return hash_code + count_collisions
	def compressor(self, hash_code):
		return hash_code % self.array_size
	def assign(self, key, value):
		index = self.compressor(self.hash(key))
		index_value = self.array[index]
		if index_value is None:
			self.array[index] = [key, value]
			return
		if index_value[0] == key:
			self.array[index] = [key, value]
			return
		count_collisions = 1
		while index_value[0] != key:
			new_index = self.compressor(self.hash(key, count_collisions))
			new_index_value = self.array[new_index]
			if new_index_value is None:
				self.array[new_index] = [key, value]
				return
			if new_index_value[0] == key:
				self.array[new_index] = [key, value]
				return
			count_collisions += 1
	def retrieve(self, key):
		index = self.compressor(self.hash(key))
		index_value = self.array[index]
		if index_value is None:
			return None
		if index_value[0] == key:
			return index_value[1]
		count_collisions = 1
		while index_value[0] != key:
			new_index = self.compressor(self.hash(key, count_collisions))
			new_index_value = self.array[new_index]
			if new_index_value is None:
				return None
			if new_index_value[0] == key:
				return self.array[1]
			count_collisions += 1

def pattern_search(text, pattern, replacement, case = True):
	fixed_text = ""
	num_skip = 0
	for i in range(len(text)):
		count = 0
		if num_skip > 0:
			num_skip -= 1
			continue
		for j in range(len(pattern)):
			if case and text[i + j] == pattern[j]:
				count += 1
			elif not case and text[i + j].lower() == pattern[j].lower():
				count += 1
			else:
				break
		if count == len(pattern):
			num_skip = len(pattern) - 1
			fixed_text += replacement
			print(f"{pattern} is found at index {str(i)}")
		else:
			fixed_text += text[i]
	return fixed_text

def bubble_sort(arr):
	for i in range(len(arr)):
		for j in range(len(arr) - i - 1):
			if arr[j] > arr[j + 1]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]
	return arr

def merge_sort(arr):
	if len(arr) <= 1:
		return arr
	middle_index = len(arr) // 2
	left_split = arr[:middle_index]
	right_split = arr[middle_index:]
	left_sorted = merge_sort(left_split)
	right_sorted = merge_sort(right_split)
	return merge(left_sorted, right_sorted)

def merge(left, right):
	result  = []
	while left and right:
		if left[0] < right[0]:
			result.append(left[0])
			left.pop(0)
		else:
			result.append(right[0])
			right.pop(0)
	if left:
		result += left
	if right:
		result += right
	return result

from random import randrange
def quicksort(arr, start, end):
	if start >= end:
		return
	pivot_idx = randrange(start, end + 1)
	pivot_element = arr[pivot_idx]
	arr[end], arr[pivot_idx] = arr[pivot_idx], arr[end]
	pointer = start
	for i in range(start, end):
		if arr[i] < pivot_element:
			arr[i], arr[pointer] = arr[pointer], arr[i]
			pointer += 1
	arr[end], arr[pointer] = arr[pointer], arr[end]
	quicksort(arr, start, pointer - 1)
	quicksort(arr, pointer + 1, end)

class TreeNode:
	def __init__(self, value):
		self.value = value
		self.children = []
	def add_child(self, child_node):
		self.children.append(child_node)
	def remove_child(self, child_node):
		self.children = [child for child in self.children if child is not child_node]
	def traverse(self):
		nodes_to_visit = [self]
		while nodes_to_visit:
			current_node = nodes_to_visit.pop()
			print(current_node.value)
			nodes_to_visit += current_node.children

from collections import deque
def bfs(root_node, goal_value):
	path_queue = deque()
	initial_path = [root_node]
	path_queue.appendleft(initial_path)
	while path_queue:
		current_path = path_queue.pop()
		current_node = current_path[-1]
		if current_node.value == goal_value:
			for node in current_path:
				print(node.value)
			return
		for child in current_node.children:
			new_path = current_path[:]
			new_path.append(child)
			path_queue.appendleft(new_path)
	print("This path does not exist!")

def dfs(root_node, goal_value, path = ()):
	path = path + (root_node,)
	if root_node.value == goal_value:
		for node in path:
			print(node.value)
	for child in root_node.children:
		path_found = dfs(child, goal_value, path)
		if path_found is not None:
			return path_found
	return None

def recursive_binary_search(sorted_lst, left_pointer, right_pointer, target):
	if left_pointer >= right_pointer:
		return "Value not found"
	middle_idx = (left_pointer + right_pointer) // 2
	middle_value = sorted_lst[middle_idx]
	if middle_value == target:
		return middle_idx
	if middle_value > target:
		return recursive_binary_search(sorted_lst, left_pointer, middle_idx, target)
	if middle_value < target:
		return recursive_binary_search(sorted_lst, middle_idx + 1, right_pointer, target)

def iterative_binary_search(sorted_lst, target):
	left_pointer = 0
	right_pointer = len(sorted_lst)
	while left_pointer < right_pointer:
		middle_idx = (left_pointer  + right_pointer) // 2
		middle_value = sorted_lst[middle_idx]
		if middle_val == target:
			return middle_idx
		if target < middle_value:
			right_pointer = middle_idx
		if target > middle_value:
			left_pointer = middle_idx + 1
	return "Value not found"

class BinarySearchTree:
	def __init__(self, value, depth = 1):
		self.value = value
		self.depth = depth
		self.left = None
		self.right = None
	def insert(self, value):
		if value < self.value:
			if self.left is None:
				self.left = BinarySearchTree(value, self.depth + 1)
			else:
				self.left.insert(value)
		else:
			if self.right is None:
				self.right = BinarySearchTree(value, self.depth + 1)
			else:
				self.right.insert(value)
	def get_node_by_value(self, value):
		if value == self.value:
			return self
		elif self.left is not None and value < self.value:
			return self.left.get_node_by_value(value)
		elif self.right is not None and value >= self.value:
			return self.right.get_node_by_value(value)
		else:
			return None
	def depth_first_transversal(self):
		if self.left is not None:
			self.left.depth_first_traversal()
		if self.right is not None:
			self.right.depth_first_traversal()