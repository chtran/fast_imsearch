import fnmatch
import glob
import numpy as np
import os
import time

from PIL import Image
from scipy.io import loadmat
from vptree import *

TRAIN_DIR = '/data/hays_lab/people/hays/baikal/flickr_final_new_descs_fixed/Barcelona'
TEST_DIR = '/data/hays_lab/people/hays/baikal/flickr_final_new_descs_fixed/Madrid/00001'

def l2_distance(a, b):
    a, b = a.feature, b.feature
    return np.linalg.norm(a-b)

def default_get_node(mat_file):
    file_content = loadmat(mat_file)
    gist_matrix = file_content['gist_gabor_4x4']
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
        print "Found %d files. Elapsed: %s" % (len(matches), time.time() - self.start_time)
        nodes = [self.get_node(f) for f in matches]
        self.tree = VPTree(nodes, self.distance)
        print "Done constructing tree. Elapsed: %s" % (time.time() - self.start_time)

    def get_nearest_neighbors(self, mat_file, k):
        node = self.get_node(mat_file)
        neighbors = self.tree.get_nearest_neighbors(
                node, 5)
        print "Done searching. Elapsed: %s" % (time.time() - self.start_time)
        return neighbors

if __name__ == '__main__':
    tree = ImageTree(TRAIN_DIR)
    mat_files = glob.glob(TEST_DIR + '/*.mat')
    test_file = mat_files[2]
    im = Image.open(get_img_from_mat(test_file))
    im.show()
    for result in tree.get_nearest_neighbors(test_file, 3):
        print result
        path = get_img_from_mat(result[1].path)
        im_result = Image.open(path)
        im_result.show()
