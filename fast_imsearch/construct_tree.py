import fnmatch
import glob
import numpy as np
import os
import time

from PIL import Image
from scipy.io import loadmat
from vptree import *
from fast_imsearch.lshash import LSHash

TRAIN_DIR = '/data/hays_lab/people/hays/baikal/flickr_final_new_descs_fixed/Barcelona'
TEST_DIR = '/data/hays_lab/people/hays/baikal/flickr_final_new_descs_fixed/Madrid/00001'

def l2_distance(a, b):
    a, b = a.feature, b.feature
    return np.linalg.norm(a-b)

def default_get_node(mat_file):
    file_content = loadmat(mat_file)
    gist_matrix = file_content['color_hist_geom_weighted']
    return Node(np.ndarray.flatten(gist_matrix), mat_file)

def get_img_from_mat(mat_file):
    arr = mat_file.split('/')
    arr[6] = 'flickr_geo_and_gps'
    arr[-1]= arr[-1][:-3] + 'jpg'
    return '/'.join(arr)

class Node:
    def __init__(self, feature, path=None):
        self.path = path
        self.feature = feature

class ImageTree(object):
    def __init__(self, 
            input_dir,
            distance=l2_distance, 
            get_node=default_get_node):

        self.start_time = time.time()
        self.distance = distance
        self.get_node = get_node

        self._create_from_dir_recur(input_dir)

    def _create_from_dir_recur(self, input_dir):
        print "Start searching for mat files in", input_dir
        matches = []
        for root, dirnames, filenames in os.walk(input_dir):
            for filename in fnmatch.filter(filenames, '*.mat'):
                matches.append(os.path.join(root, filename))
        print "Found %d files." %  len(matches)
        nodes = [self.get_node(f) for f in matches]
        self.tree = VPTreeSplit(nodes, self.distance)
        print "Done constructing tree. Elapsed: %s" % (time.time() - self.start_time)

    def get_nearest_neighbors(self, mat_file, k):
        node = self.get_node(mat_file)
        neighbors = self.tree.get_nearest_neighbors(
                node, 5)
        print "Done searching. Elapsed: %s" % (time.time() - self.start_time)
        return neighbors

class ImageLSH(object):
    def __init__(self, input_dir, get_node=default_get_node):
        self.last_time = time.time()
        self.get_node = get_node
        self._create_from_dir_recur(input_dir)

    def record(self):
        last = self.last_time
        self.last_time = time.time()

        print "Spent:",self.last_time - last

    def _create_from_dir_recur(self, input_dir):
        print "Start searching for mat files in", input_dir
        matches = []
        for root, dirnames, filenames in os.walk(input_dir):
            for filename in fnmatch.filter(filenames, '*.mat'):
                matches.append(os.path.join(root, filename))
        print "Found %d files."
        self.record()
        self.lsh = LSHash(8, 784)
        for f in matches:
            n = self.get_node(f)
            self.lsh.index(n.feature, extra_data=n.path)
        print "Done constructing tree."
        self.record()

    def get_nearest_neighbors(self, mat_file, k):
        node = self.get_node(mat_file)
        return self.lsh.query(node.feature, num_results=k)
        print "Done searching. Elapsed: %s" % (time.time() - self.start_time)


if __name__ == '__main__':
    lsh = ImageLSH(TRAIN_DIR)
    mat_files = glob.glob(TEST_DIR + '/*.mat')
    test_file = mat_files[2]
    im = Image.open(get_img_from_mat(test_file))
    im.show()
    for result in lsh.get_nearest_neighbors(test_file, 3):
        path = get_img_from_mat(result[0][1])
        print path
        im_result = Image.open(path)
        im_result.show()
