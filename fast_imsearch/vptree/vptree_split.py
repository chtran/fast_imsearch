import random
import numpy as np
import heapq
import multiprocessing as mp
import threading
import fast_imsearch

from collections import namedtuple
from collections import deque


LEAF_SIZE = 16
SPLIT_SIZE = 16

class VPTreeSplit(object):
    """
    An efficient data structure to perform nearest-neighbor
    search. 
    """

    def __init__(
            self, points, dist_fn,
            leaf_size=LEAF_SIZE, split_size=SPLIT_SIZE):
        self.children = None
        self.upper= None
        self.lower= None
        self.dist_fn = dist_fn
        self.children = None
        self.leaf_points = None

        # choose a better vantage point selection process
        self.vp = fast_imsearch.vptree.select_vp(points, dist_fn)

        if len(points) < 1:
            return
        if len(points) <= LEAF_SIZE:
            self.leaf_points = points
            return

        # choose division boundary at median of distances
        distances = [(p, self.dist_fn(self.vp, p)) for p in points]
        sorted_distances = sorted(distances, key=lambda x:x[1])
        child_size = len(points) / split_size + 1
        
        self.lower = []
        self.upper = []
        self.children = []
        for i in range(split_size):
            lo = i * child_size
            hi = min((i+1) * child_size - 1, len(points)-1)
            child_points = [t[0] for t in sorted_distances[lo:hi+1]]
            if len(child_points) == 0:
                break
            self.lower.append(sorted_distances[lo][1])
            self.upper.append(sorted_distances[hi][1])
            self.children.append(VPTreeSplit(
                points=child_points,
                dist_fn=dist_fn,
                leaf_size=leaf_size,
                split_size=split_size))

    def is_leaf(self):
        return self.leaf_points is not None

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
                tau, _ = neighbors.queue[-1]

            if node.is_leaf():
                total_seen += len(node.leaf_points)
                for child_p in node.leaf_points:
                    child_d = self.dist_fn(q, child_p)
                    neighbors.push(child_d, child_p)
                tau, _ = neighbors.queue[-1]
                continue

            if node.children is None:
                continue
            for i,mu in enumerate(node.upper):
                if d < mu:
                    if d + tau >= node.lower[i]:
                        self.add_node(visit_stack, node.children[i],q)
                else:
                    if d - tau <= node.lower[i]:
                        self.add_node(visit_stack, node.children[i],q)

        print "number nodes seen:",total_seen
        return neighbors.queue

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
