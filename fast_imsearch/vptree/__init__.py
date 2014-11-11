from collections import namedtuple
from collections import deque
import random
import numpy as np
import heapq
import multiprocessing as mp
import threading

MIN_LEAF = 100

def _create_sub_tree(tree, distances, points, is_left, level):
    sub_tree = None
    if len(points) > 0:
        sub_tree = VPTree(
                points=points, dist_fn=tree.dist_fn, level=level)
    if is_left:
        tree.left = sub_tree
    else:
        tree.right = sub_tree

class SubTreeThread(threading.Thread):
    def __init__(self, tree, distances, points, is_left, level):
        threading.Thread.__init__(self)
        self.tree = tree
        self.distances = distances
        self.points = points
        self.is_left = is_left
        self.level = level

    def run(self):
        _create_sub_tree(
            self.tree, self.distances, 
            self.points, self.is_left, self.level)

class VPTree(object):
    """
    An efficient data structure to perform nearest-neighbor
    search. 
    """

    def __init__(
            self, points, dist_fn, min_leaf=MIN_LEAF, level=0):
        self.left = None
        self.right = None
        self.mu = None
        self.dist_fn = dist_fn

        if len(points) < min_leaf:
            self.all_points = points
            return

        # choose a better vantage point selection process
        self.vp = points.pop(random.randrange(len(points)))

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
        if level < 3:
            left_thread = SubTreeThread(
                    self, distances, left_points, True, level+1)
            right_thread = SubTreeThread(
                    self, distances, right_points, False, level+1)
            left_thread.start()
            right_thread.start()
            left_thread.join()
            right_thread.join()
        else:
            _create_sub_tree(
                self, distances, left_points, True, level+1)
            _create_sub_tree(
                self, distances, right_points, False, level+1)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

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
        visit_stack = deque([self])

        # distance of n-nearest neighbors so far
        tau = np.inf

        while len(visit_stack) > 0:
            node = visit_stack.popleft()
            if node is None:
                continue

            if node.is_leaf():
                for point in node.all_points:
                    neighbors.push(self.dist_fn(q, point), point)
                continue

            d = self.dist_fn(q, node.vp)
            if d < tau:
                neighbors.push(d, node.vp)
                tau, _ = neighbors.queue[-1]

            if d < node.mu:
                visit_stack.append(node.left)
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
            else:
                visit_stack.append(node.right)
                if d < node.mu + tau:
                    visit_stack.append(node.left)
        return neighbors.queue


    def get_all_in_range(self, q, tau):
        """
        find all points within a given radius of point q

        :param q: a query point
        :param tau: the maximum distance from point q
        """

        # buffer for nearest neightbors
        neighbors = []

        # list of nodes ot visit
        visit_stack = deque([self])

        while len(visit_stack) > 0:
            node = visit_stack.popleft()
            if node is None:
                continue

            d = self.dist_fn(q, node.vp)
            if d < tau:
                neighbors.append((d, node.vp))

            if node.is_leaf():
                continue

            if d < node.mu:
                if d < node.mu + tau:
                    visit_stack.append(node.left)
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
            else:
                if d >= node.mu - tau:
                    visit_stack.append(node.right)
                if d < node.mu + tau:
                    visit_stack.append(node.left)
        return neighbors

class PriorityQueue(object):
    def __init__(self, size=None):
        self.queue = []
        self.size = size

    def push(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()
        if self.size is not None and len(self.queue) > self.size:
            self.queue.pop()
