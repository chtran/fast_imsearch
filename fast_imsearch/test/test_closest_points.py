import numpy as np
import fast_imsearch
from fast_imsearch.vptree import *

if __name__ == '__main__':
    X = np.random.uniform(0, 100000, size=10000)
    Y = np.random.uniform(0, 100000, size=10000)
    points = [NDPoint(x,i) for i, x in  enumerate(zip(X,Y))]
    tree = VPTree(points)
    q = NDPoint([300,300])
    neighbors = tree.get_nearest_neighbors(q, 5)

    print "query:"
    print "\t", q
    print "nearest neighbors: "
    for d, n in neighbors:
        print "\t", n
