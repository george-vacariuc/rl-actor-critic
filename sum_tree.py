import numpy as np

class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, np.float64)
        self.p_min = 1e5

    def add(self, priority):
        assert priority > 0
        self.p_min = min(self.p_min, priority)
        tree_index = self.data_pointer + self.capacity - 1
        insertion_pos = self.data_pointer

        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        return insertion_pos

    def update(self, tree_index, priority):
        assert priority > 0
        self.tree[tree_index] = priority

        while tree_index > 0:
            tree_index = (tree_index - 1) // 2
            prev = self.tree[tree_index]
            self.tree[tree_index] = self.sum_children(tree_index)

    def sum_children(self, parent):
        left = 2 * parent + 1
        right = left + 1
        return self.tree[left] + self.tree[right]

    def get_leaf(self, v):
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]
                parent = right

        data_index = leaf - self.capacity + 1
        return leaf, self.tree[leaf]

    @property
    def total_priority(self):
        return self.tree[0]
