import numpy as np
import time
from fast_imsearch.lshash import LSHash

def l2(x,y):
    return np.linalg.norm(x-y)

def brute_force(train_points, test_point, k):
    distances = [(p, l2(test_point, p)) for p in train_points]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    return sorted_distances[:k]

if __name__ == "__main__":
    num_dimension = 2000
    num_samples = 100000
    num_test = 10
    k = 5
    X = np.random.uniform(0, 100, size=(num_samples, num_dimension))
    Y = np.random.uniform(0, 100, size=(num_test, num_dimension))
    train_points = [X[i,:] for i in range(num_samples)]
    lsh = LSHash(8, num_dimension)
    start_cons = time.time()
    for i in range(num_samples):
        lsh.index(X[i,:])
    print "done construction in", time.time() - start_cons
    for i in range(num_test):
        test_point = Y[i,:]
        start_time = time.time()
        lsh_neighbors = lsh.query(test_point, num_results=k, distance_func='true_euclidean')
        done_lsh = time.time()
        brute_force_neighbors = brute_force(train_points, test_point, k)
        done_brute_force = time.time()
        print "lsh in:", done_lsh-start_time
        print "brute-force in:", done_brute_force-done_lsh
        assert len(lsh_neighbors) == k
        assert len(brute_force_neighbors) == k
