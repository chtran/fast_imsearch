import glob
import numpy as np

from PIL import Image
from scipy.io import loadmat
from vptree import *

TRAIN_DIR = '/data/hays_lab/people/hays/baikal/flickr_final_new_descs_fixed/Barcelona/00001'
TEST_DIR = '/data/hays_lab/people/hays/baikal/flickr_final_new_descs_fixed/Madrid/00001'
MAX_CHILDREN = 100

def l2_distance(a, b):
    a, b = a.feature, b.feature
    assert len(a) == len(b)
    return np.sum([np.power(a[i] - b[i], 2) \
            for i in xrange(len(a))])

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

        self.distance = distance
        self.get_node = get_node

        self._create_from_dir(input_dir)

    def _create_from_dir(self, input_dir):
        mat_files = glob.glob(input_dir + '/*.mat')
        print "Found %d files" % len(mat_files)
        features = [self.get_node(f) for f in mat_files]
        self.tree = VPTree(features, self.distance, MAX_CHILDREN)
        return self.tree

    def get_nearest_neighbors(self, mat_file):
        yield list(self.tree.find(self.get_node(mat_file)))

if __name__ == '__main__':
    tree = ImageTree(TRAIN_DIR)
    mat_files = glob.glob(TEST_DIR + '/*.mat')
    test_file = mat_files[2]
    n = 0
    im = Image.open(get_img_from_mat(test_file))
    im.show()
    for result in tree.get_nearest_neighbors(test_file):
        path = get_img_from_mat(result[0][0].path)
        im_result = Image.open(path)
        im_result.show()
        n += 1
        if n >= 5: break
