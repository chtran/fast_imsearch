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
    num_samples = 100000
    num_dimension = 1000
    num_test = 3
    k = 100

    X = np.random.uniform(0, 100000, size=(num_samples, num_dimension))
    Y = np.random.uniform(0, 100000, size=(num_test, num_dimension))
    points = [NDPoint(X[i,:], i) for i in range(np.size(X,0))]
    start_time = time.time()
    print "Start constructing single threaded tree"
    single_tree = VPTree(points, l2)
    done_single = time.time()
    print "Constructed single tree in", done_single - start_time

    #print "Start constructing parallel tree"
    #parallel_tree = ParallelVPTree(4, points, l2)
    #print "Constructed parallel tree in", time.time() - done_single
    test_points = [NDPoint(Y[i,:], i) for i in range(np.size(Y,0))]
    for i, q in enumerate(test_points):
        print
        print "Test #",i
        start_time = time.time()
        single_neighbors = single_tree.get_nearest_neighbors(q, k)
        done_single = time.time()
        #parallel_neighbors = parallel_tree.get_nearest_neighbors(q, k)
        #done_parallel = time.time()
        brute_force_neighbors = brute_force(q, points, k)
        done_brute_force = time.time()
        print "Single:", done_single - start_time
        #print "Multiple:", done_parallel - done_single
        print "Brute-force:", done_brute_force - done_single
        for i in range(k):
            assert single_neighbors[i][1].idx == brute_force_neighbors[i][1].idx
        #    assert parallel_neighbors[i][1].idx == brute_force_neighbors[i][1].idx
    print "Success"
