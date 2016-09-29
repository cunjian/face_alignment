# coding: utf-8

import numpy as np
import cv2
import os
import caffe
import sys
from numpy.linalg import norm
from predict import Predict, load_test_img
from common.get_data import load_celeba_data, get_train_val_test_list, getDataFromTXT
from common.utils import check_bbox

def compute_landmark_norm(landmarkP, landmarkR, bbox):
	error = np.zeros(5)
	for i in range(5):
		error[i] = norm(landmarkP[i] - landmarkR[i])
	error = error / bbox.w
	return error

def compute_error(data, model):
	error = np.zeros((len(data), 5))
	net = Predict('../deploy.prototxt', model)
	for i, (imgpath, bbox, landmarkR) in enumerate(data):
		gray = cv2.imread(imgpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		if not check_bbox(gray, bbox):
			print 'BBox out of range...'
			continue
		face = load_test_img(gray, bbox)

		landmarkP = net.forward(face, 'fc2')
		landmarkP = bbox.reprojectLandmark(landmarkP)
		landmarkR = bbox.reprojectLandmark(landmarkR)

		print 'Computing error for %s' % imgpath
		error[i] = compute_landmark_norm(landmarkP, landmarkR, bbox)

	return error, len(data)

if __name__ == '__main__':
	# Validate for celeba dataset
	if len(sys.argv) > 2 and sys.argv[2] == 'celeba':
		print 'Loading validation data...'
		data = load_celeba_data()
		train, val, test = get_train_val_test_list()
		val_data = [data[i-1] for i in val]

	# Validate for original dataset
	else:
		valid_txt = '../dataset/train/testImageList.txt'
		val_data = getDataFromTXT(valid_txt)

	error, num = compute_error(val_data, sys.argv[1])
	error = np.mean(error, axis=0)
	print 'Validation for %s images' % len(val_data) 
	print '************** Mean Error of %d images **************' % num
	print 'Lefteye error:		%f' % error[0]
	print 'Righteye error:		%f' % error[1]
	print 'Nose error:		%f' % error[2]
	print 'Leftmouth error:	%f' % error[3]
	print 'Rightmouth error:	%f' % error[4]