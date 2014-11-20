from vptree import *
from vptree_split import *
from old_implementation import *
import random


def select_vp(points, dist_fn):
    """Return the vantage point and delete it from points list."""
    #return select_median(points, dist_fn)
    return select_random(points, dist_fn)

def select_random(points, dist_fn):
    return points.pop(random.randrange(len(points)))

def select_median(points, dist_fn):
    if len(points) <= 10:
        return select_random(points, dist_fn)
    N_sample = max(10, len(points)/1000)
    N_test = max(10, len(points)/1000)
    P = random.sample(points, N_sample)
    best_spread = 0
    for p in P:
        D = random.sample(points, N_test)
        distances = [dist_fn(p, d) for d in D]
        mu = np.median(distances)
        spread = np.std(distances - mu)
        if spread > best_spread:
            best_spread = spread
            best_p = p
    points.remove(best_p)
    return best_p
