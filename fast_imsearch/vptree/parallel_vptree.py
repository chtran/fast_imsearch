from vptree import *

PARALLEL_CUTOFF = 10000
def _get_nearest_neighbors(tree, q, k):
    return tree.get_nearest_neighbors(q, k)

class ParallelVPTree(object):
    """
    Parallelize VP-tree
    """
    def __init__(
            self, num_tree, points, dist_fn, min_leaf=MIN_LEAF):
        num_points = len(points)
        if num_points < PARALLEL_CUTOFF:
            self.trees = [VPTree(points, dist_fn, min_leaf)]
        else:
            tree_size = num_points / num_tree + 1
            self.trees = []
            for i in range(num_tree):
                start_index = tree_size * i
                end_index = tree_size * (i+1)
                self.trees.append(VPTree(
                    points[start_index:end_index], 
                    dist_fn, min_leaf))

    def get_nearest_neighbors(self, q, k=1):
        num_trees = len(self.trees)
        pool = mp.Pool(processes=num_trees)
        results = [pool.apply_async(_get_nearest_neighbors, args=(tree, q, k)) for tree in self.trees]

        nearest_neighbors = [tup for l in results for tup in l.get()]
        return sorted(nearest_neighbors, key=lambda x: x[0])[:k]
        

    def get_all_in_range(self, q, tau):
        return [tup for tree in self.trees for tup in tree.get_all_in_range(q, tau)]
