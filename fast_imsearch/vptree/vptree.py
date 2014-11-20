from collections import namedtuple
from collections import deque
import random
import numpy as np
import heapq
import multiprocessing as mp
import threading

LEAF_SIZE = 16
SPLIT_SIZE = 16

class VPTree(object):
    """
    An efficient data structure to perform nearest-neighbor
    search. 
    """

    def __init__(
            self, points, dist_fn, leaf_size=LEAF_SIZE):
        self.left = None
        self.right = None
        self.mu = None
        self.dist_fn = dist_fn

        # choose a better vantage point selection process
        self.vp = fast_imsearch.vptree.select_vp(points, dist_fn)

        if len(points) < 1:
            return
        if len(points) <= LEAF_SIZE:
            self.children = points
            return

        # choose division boundary at median of distances
        distances = [self.dist_fn(self.vp, p) for p in points]
        self.mu = np.median(distances)

        left_points = []
        right_points = []
        for i, p in enumerate(points):
            d = distances[i]
            if d < self.mu:
                left_points.append(p)
            else:
                right_points.append(p)
        if len(left_points) > 0:
            self.left = VPTree(
                points=left_points, 
                dist_fn=self.dist_fn, 
                leaf_size=leaf_size)
        if len(right_points) > 0:
            self.right = VPTree(
                points=right_points, 
                dist_fn=self.dist_fn,
                leaf_size=leaf_size)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def get_size(self):
        left_size = 0
        right_size = 0
        if self.left is not None:
            left_size = self.left.get_size()
        if self.right is not None:
            right_size = self.right.get_size()
        return left_size + right_size + 1

    def get_height(self):
        left_height = 0
        right_height = 0
        if self.left is not None:
            left_height = self.left.get_height()
        if self.right is not None:
            right_height = self.right.get_height()
        return max(left_height, right_height) + 1

    def add_node(self, stack, node, q):
        if node is not None:
            d = self.dist_fn(q, node.vp)
            heapq.heappush(stack, (d, node))

    ### Operations
    def get_nearest_neighbors(self, q, k=1):
        """
        find k nearest neighbor(s) of q

        :param q: a query point
        :param k: number of nearest neighbors

        """

        # buffer for nearest neightbors
        neighbors = PriorityQueue(k)

        # list of nodes ot visit
        visit_stack = []
        self.add_node(visit_stack, self, q)

        # distance of n-nearest neighbors so far
        tau = np.inf

        total_seen = 0
        while len(visit_stack) > 0:
            d, node = heapq.heappop(visit_stack)
            total_seen += 1
            if node is None:
                continue

            if d < tau:
                neighbors.push(d, node.vp)
                if neighbors.len() == k:
                    tau, _ = neighbors.queue[-1]

            if node.is_leaf():
                total_seen += len(node.children)
                for child_p in node.children:
                    child_d = self.dist_fn(q, child_p)
                    neighbors.push(child_d, child_p)
                if neighbors.len() == k:
                    tau, _ = neighbors.queue[-1]
                continue

            if d < node.mu:
                self.add_node(visit_stack, node.left, q)
                if node.mu - d <= tau:
                    self.add_node(visit_stack, node.right, q)
            else:
                self.add_node(visit_stack, node.right, q)
                if d - node.mu <= tau:
                    self.add_node(visit_stack, node.left, q)
        print "number nodes seen:",total_seen
        return (neighbors.queue, total_seen)

class PriorityQueue(object):
    def __init__(self, size=None):
        self.queue = []
        self.size = size

    def push(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()
        if self.size is not None and len(self.queue) > self.size:
            self.queue.pop()

    def len(self):
        return len(self.queue)
