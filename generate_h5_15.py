# coding: utf-8

import numpy as np
import h5py
import sys
import os
import cv2
from os.path import exists, join, isfile
from configobj import ConfigObj
from sklearn.utils import shuffle
from common.utils import processImage, flip, BBox, check_bbox,drawLandmark
from common.get_data import load_celeba_data, get_train_val_test_list, getDataFromTXT, getDataFromTXT_68, getDataFromTXT_15

def generate_h5py(data, h5_path, txt_path, augment=False):
	'''
	Get images and turn them into h5py files
	Input:
		- data: a tuple of [imgpath, bbox, landmark]
		- h5_path: h5py file name
		- txt_path: h5 txt name
	'''
	F_imgs = []
	F_landmarks = []
	num_sample=1
	for (imgpath, bbox, landmark) in data:
		img = cv2.imread(imgpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		print ("processing %s") % imgpath
		if not check_bbox(img, bbox): 
			print 'BBox out of range.'
			continue
		face = img[bbox.top:bbox.bottom, bbox.left:bbox.right]
		# debug
		landmark = bbox.reprojectLandmark(landmark)
		img = drawLandmark(img, bbox, landmark)
		fp = 'debug_results/'+ str(num_sample)+'.jpg'
		cv2.imwrite(fp, img)
		num_sample=num_sample+1

		if augment:
			face_flip, landmark_flip = flip(face, landmark)
			face_flip = cv2.resize(face_flip, (39,39)).reshape(1,39,39)
			landmark_flip = landmark_flip.reshape(30)
			F_imgs.append(face_flip)
			F_landmarks.append(landmark_flip)

		face = cv2.resize(face, (39,39)).reshape(1,39,39)
		landmark = landmark.reshape(30)
		F_imgs.append(face)
		F_landmarks.append(landmark)

	F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
	F_imgs = processImage(F_imgs)
	F_imgs, F_landmarks = shuffle(F_imgs, F_landmarks, random_state=42)
	
	with h5py.File(h5_path, 'w') as f:
		f['data'] = F_imgs.astype(np.float32)
		f['landmarks'] = F_landmarks.astype(np.float32)
	with open(txt_path, 'w') as f:
		f.write(h5_path)
		f.write(str(num_sample))

if __name__ == '__main__':
	config = ConfigObj('config_300w.txt')
	train_txt = config['train_txt']
	valid_txt = config['valid_txt']
	h5_dir = 'h5_15'
	if not exists(h5_dir):
		os.mkdir(h5_dir)
	#assert(isfile('config.txt') and isfile(train_txt) and isfile(valid_txt) and exists(h5_dir))

	if len(sys.argv) > 1 and sys.argv[1] == 'celeba':
		data = load_celeba_data()
		train_list, val_list, test_list = get_train_val_test_list()
		train_data = [data[i-1] for i in train_list]
		val_data = [data[i-1] for i in val_list]

		generate_h5py(train_data, join(h5_dir,'train.h5').replace('\\','/'), join(h5_dir,'train.txt'), augment=True)
		generate_h5py(val_data, join(h5_dir,'valid.h5').replace('\\','/'), join(h5_dir,'valid.txt'), augment=True)

	else:
		train_data = getDataFromTXT_15(train_txt)
		val_data = getDataFromTXT_15(valid_txt)

		generate_h5py(train_data, join(h5_dir,'train.h5').replace('\\','/'), join(h5_dir,'train.txt'), augment=True)
		generate_h5py(val_data, join(h5_dir,'valid.h5').replace('\\','/'), join(h5_dir,'valid.txt'), augment=True)
