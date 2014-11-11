import numpy as np
import time
import fast_imsearch
from fast_imsearch.vptree import *

class NDPoint(object):
    """
    A point in n-dimensional space
    """

    def __init__(self, x, idx=None):
        self.x = np.array(x)
        self.idx = idx
    def __repr__(self):
        return "NDPoint(idx=%s, x=%s)" % (self.idx, self.x)

### Distance functions
def l2(p1, p2):
    return np.linalg.norm(p1.x-p2.x)

def brute_force(q, points, k):
    distances = [(l2(p,q), p) for p in points]
    sorted_distances = sorted(distances, key=lambda x: x[0])
    return sorted_distances[:k]

if __name__ == '__main__':
    num_samples = 500000
    num_dimension = 1000
    num_test = 3
    k = 100

    X = np.random.uniform(0, 100000, size=(num_samples, num_dimension))
    Y = np.random.uniform(0, 100000, size=(num_test, num_dimension))
    points = [NDPoint(X[i,:], i) for i in range(np.size(X,0))]
    start_time = time.time()
    print "Start constructing tree"
    tree = VPTree(points, l2)
    print "Constructed tree in", time.time() - start_time
    test_points = [NDPoint(Y[i,:], i) for i in range(np.size(Y,0))]
    for q in test_points:
        start_time = time.time()
        neighbors = tree.get_nearest_neighbors(q, k)
        done_tree = time.time()
        brute_force_neighbors = brute_force(q, points, k)
        done_brute_force = time.time()
        print "tree: %f. brute-force: %f" % (done_brute_force - done_tree, done_tree - start_time)
        for i in range(k):
            assert neighbors[i][1].idx == brute_force_neighbors[i][1].idx
    print "Success"
