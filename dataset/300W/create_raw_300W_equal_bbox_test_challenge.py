"""
This module creates the 68 keypoint 300W datasets (as pickle files) by using the orginal images in 'src_dir'. It keeps the images in their
RGB format but with a reduced_size.

#################################

Here is the instruction to create 300W datasets:

1 - Download Helen, LFPW, AFW and IBUG datasets from:
http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

2 - Once unzipped, the helen and lfpw have two subdirectories, 'testset' and 'trainset'.
Rename them to 'X_testset' and 'X_trainset', for each dataset X.

3 - Create one directory named 'Train_set' and put unzipped 'afw', 'helen_trainset'
and 'lfpw_trainset' directories into it (as three sub-directories).

4 - Create another directory named 'Test_set' and put unzipped 'ibug', 'helen_testset' and 'lfpw_testset'
into it (as three sub-directories).

5 - Put Train_set and Test_set directories into one direcotory (i.e. 300W) and pass
the complete path to it to 'src_dir' when calling this module.

6 - Call create_raw_300W.py module by passing complete path to 'src_dir' and 'dest_dir' arguments:
python create_raw_300W.py --src_dir=/complete/path/to/300W/folder --dest_dir=/complete/path/to/RCN/datasets

**Note: dest_dir is the location where the dataset will be created. It should be finally put in RCN/datasets directory
of the repo

This module will create 300W_test_160by160.pickle and 300W_train_160by160.pickle files in the given dest_dir path.

**Note: follow the instructions here for citation if you use 300W dataset:
http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

#################################

Note: This module creates a dataset with 3148 train images (afw, Helen, lfpw), and 689 test images (135 ibug, 330 Helen, 224 lfpw)
with 68 keypoints.

The process in this module is as follows:
For all images in the train and test sets, a rectangle around face is detected.
the face is cropped within the detected rectangle and then downsampled to reduced_size.
In order to avoid aliasing while downsampling, Image library with Image.ANTIALIAS downsampling feature is used.
The downsampled Image is then converted from PIL to BGR format, which is the default format in cv2 images.
Then the original keypoint locations, the detected rectangle, and the normalized key-point locations
in the range [0, 1] is kept. Note that the normalized keypoint locations can be multiplied in the reduced_size (the image size) to get
the pixel locations.

All other pre-processings (detection of the bounding box, gray-scaling, downsampling, and contrast normalization)
are post-poned to later stages.

The created datasets are in two files, MTFL_test and MTFL_train. Each file contains an orderedDict with 'X' and 'Y' label.
'X' is a 4D tensor of size (#sampels, #rows, #cols, #channels). Each image sample is in cv2's BGR format
'Y' is an orderedDict with the following components. Each conponent has as many rows as the number of samples.
'kpt_orig'    : the original position of the keypoints in the format
                x1 y1 ... x5 y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
'kpt_norm'    : the normalized keypoint positions such that each x or y is in the range of [0,1]
                key_point_x_normalized = ( keypoint_x - rect_start_x ) / rect_width
'face_rect'   : a four value vector of values: rect_start_x, rect_start_y, rect_width (rect_end_x - rect_start_x), rect_height (rect_end_y - rect_start_y)
"""

import cv2
import cv
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
import cPickle as pickle
import string
import copy
import argparse
import dlib
import math

class BBox:  # Bounding box
    
    @staticmethod
    def BBoxFromLTRB(l, t, r, b):
        return BBox(l, t, r, b)

    def top_left(self):
        return (self.top, self.left)
    
    def left_top(self):
        return (self.left, self.top)

    def bottom_right(self):
        return (self.bottom, self.right)
    
    def right_top(self):
        return (self.right, self.top)

    def __init__(self,left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

def get_kpts(file_path):
    kpts = []
    f = open(file_path, 'r')
    ln = f.readline()
    while not ln.startswith('n_points'):
        ln = f.readline()

    num_pts = ln.split(':')[1]
    num_pts = string.strip(num_pts)
    # checking for the number of keypoints
    if float(num_pts) != num_kpts:
        print "encountered file with less than %f keypoints in %s" %(num_kpts, file_pts)
        return None

    # skipping the line with '{'
    ln = f.readline()

    ln = f.readline()
    while not ln.startswith('}'):
        vals = ln.split(' ')[:2]
        vals = map(string.strip, vals)
        vals = map(np.float32, vals)
        kpts.append(vals)
        ln = f.readline()
    return kpts


if __name__ == "__main__":
    # the path to the downsampled directory
    parent_dir = '/home/cunjian/code/caffe/examples/DeepFace-master/FaceAlignment/DataPrecessing/300-W'
    out_dir = '/home/cunjian/code/caffe/examples/VanillaCNN-master/FaceAlignment/DataPrecessing/300-W'

    # global parameters
    num_kpts = 68
    text_file = open("300W_testset_challenge.txt", "w")

    # Read lfpw dataset
    train_dir_lfpw = "%s/ibug" %(parent_dir)
    #test_dir = "%s/helen/testset" %(parent_dir)

    # read pts file
    path_lfpw = "%s" %(train_dir_lfpw)
    files_lfpw = os.listdir(path_lfpw)
    files_lfpw = [i for i in files_lfpw if i.endswith('.pts')]

    for index, file_pts in enumerate(files_lfpw):
        file_path = "%s/%s" %(path_lfpw, file_pts)
        kpts =  get_kpts(file_path)
        
        if kpts is None:
            continue

        file_jpg = file_pts.split('.')[0] + '.jpg'
        jpg_path =  "%s/%s" %(path_lfpw, file_jpg)
        if not os.path.isfile(jpg_path):
            file_jpg = file_pts.split('.')[0] + '.png'
            jpg_path =  "%s/%s" %(path_lfpw, file_jpg)
        
        # convert GT points to an array
        GT_points = np.asarray(kpts)

        # obtain bounding box position for entire set of landmarks, defined as the enclosing box 
        # the bounding box provided by 
        #left_boxface,  top_boxface = GT_points.min( axis=0 )
        #right_boxface, bot_boxface = GT_points.max( axis=0 )

	# crop face box
        x_min, y_min = GT_points.min(0)
        x_max, y_max = GT_points.max(0)
        w, h = x_max-x_min, y_max-y_min
        w = h = min(w, h)
        ratio = 0.1
        x_new = x_min - w*ratio
        y_new = y_min - h*ratio
        w_new = w*(1 + 2*ratio)
        h_new = h*(1 + 2*ratio)
        bbox = map(int, [x_new, x_new+w_new, y_new, y_new+h_new])
        bbox = BBox(bbox)
        left_boxface=x_new
        right_boxface=x_new+w_new
        top_boxface=y_new
        bot_boxface=y_new+h_new

        # output jpg_path and the landmark points
        text_file.write("%s" % jpg_path)
        # output bbox
        text_file.write(" %s %s %s %s" % (str(int(math.floor(left_boxface))),str(int(math.ceil(right_boxface))),str(int(math.floor(top_boxface))), str(int(math.ceil(bot_boxface)))))

        #for x, y in kpts:
        #    text_file.write(" %s %s" % (str(x), str(y)))
        point_list=range(0,68)
        for index in point_list:
            text_file.write(" %s %s" % (kpts[index][0], kpts[index][1]))
        text_file.write('\n')



 
