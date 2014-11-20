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
    dimensions = [20,50, 100]
    for num_dimension in dimensions:
        print
        print "Running d=",num_dimension
        num_samples = 200000
        num_test = 20
        k = 1

        X = np.random.uniform(0, 100, size=(num_samples, num_dimension))
        Y = np.random.uniform(0, 100, size=(num_test, num_dimension))
        points = [NDPoint(X[i,:], i) for i in range(np.size(X,0))]
        start_time = time.time()
        single_tree = VPTreeSplit(points, l2)
        done_single = time.time()
        print "Constructed single tree in", done_single - start_time

        #print "Start constructing parallel tree"
        #parallel_tree = ParallelVPTree(4, points, l2)
        #print "Constructed parallel tree in", time.time() - done_single
        test_points = [NDPoint(Y[i,:], i) for i in range(np.size(Y,0))]
        nodes_seen = []
        for i, q in enumerate(test_points):
            print
            print "Test #",i
            start_time = time.time()
            single_neighbors, seen = single_tree.get_nearest_neighbors(q, k)
            nodes_seen.append(seen)
            done_single = time.time()
            #parallel_neighbors = parallel_tree.get_nearest_neighbors(q, k)
            #done_parallel = time.time()
            brute_force_neighbors = brute_force(q, points, k)
            done_brute_force = time.time()
            print "Single:", done_single - start_time
            #print "Multiple:", done_parallel - done_single
            print "Brute-force:", done_brute_force - done_single
            assert len(single_neighbors) == k
            for i in range(k):
                assert single_neighbors[i][1].idx == brute_force_neighbors[i][1].idx
            #    assert parallel_neighbors[i][1].idx == brute_force_neighbors[i][1].idx
        print "mean nodes seen:", np.mean(nodes_seen)
        print "Success"
