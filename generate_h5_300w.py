# coding: utf-8

import numpy as np
import h5py
import sys
import os
import cv2
from os.path import exists, join, isfile
from configobj import ConfigObj
from sklearn.utils import shuffle
from common.utils import processImage, flip, BBox, check_bbox, rotate
from common.get_data import load_celeba_data, get_train_val_test_list, getDataFromTXT, getDataFromTXT_68, getDataFromTXT_15, getDataFromTXT_5

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
	img_size=227
	num_landmark=15*2
	for (imgpath, bbox, landmarkGt) in data:
		img = cv2.imread(imgpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		print ("processing %s") % imgpath
		if not check_bbox(img, bbox): 
			print 'BBox out of range.'
			continue
		face = img[bbox.top:bbox.bottom, bbox.left:bbox.right]

		if augment:
			# flip the face
			face_flip, landmark_flip = flip(face, landmarkGt)
			face_flip = cv2.resize(face_flip, (img_size,img_size)).reshape(1,img_size,img_size)
			landmark_flip = landmark_flip.reshape(num_landmark)
			F_imgs.append(face_flip)
			F_landmarks.append(landmark_flip)
			#print landmark_flip

			# rotate the face
			face_rotated_by_alpha, landmark_rotated_5 = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), 5)
			landmark_rotated_5 = bbox.projectLandmark(landmark_rotated_5)
			face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))
			F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_rotated_5.reshape(num_landmark))
			#print landmark_rotated_5
			# flip with the rotation
			face_flipped, landmark_flipped_5 = flip(face_rotated_by_alpha, landmark_rotated_5)
			face_flipped = cv2.resize(face_flipped, (img_size,img_size))
			F_imgs.append(face_flipped.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_flipped_5.reshape(num_landmark))
			#print landmark_flipped_5

			# rotate the face
			face_rotated_by_alpha, landmark_rotated_10 = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), 10)
			landmark_rotated_10 = bbox.projectLandmark(landmark_rotated_10)
			face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))
			F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_rotated_10.reshape(num_landmark))
			#print landmark_rotated_10
			# flip with the rotation
			face_flipped, landmark_flipped_10 = flip(face_rotated_by_alpha, landmark_rotated_10)
			face_flipped = cv2.resize(face_flipped, (img_size,img_size))
			F_imgs.append(face_flipped.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_flipped_10.reshape(num_landmark))
			#print landmark_flipped_10

			# rotate the face
			face_rotated_by_alpha, landmark_rotated_15 = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), 15)
			landmark_rotated_15 = bbox.projectLandmark(landmark_rotated_15)
			face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))
			F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_rotated_15.reshape(num_landmark))
			#print landmark_rotated_15
			# flip with the rotation
			face_flipped, landmark_flipped_15 = flip(face_rotated_by_alpha, landmark_rotated_15)
			face_flipped = cv2.resize(face_flipped, (img_size,img_size))
			F_imgs.append(face_flipped.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_flipped_15.reshape(num_landmark))
			#print landmark_flipped_15

			# rotate the face
			face_rotated_by_alpha, landmark_rotated_20 = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), 20)
			landmark_rotated_20 = bbox.projectLandmark(landmark_rotated_20)
			face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))
			F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_rotated_20.reshape(num_landmark))
			#print landmark_rotated_20

			# flip with the rotation
			face_flipped, landmark_flipped_20 = flip(face_rotated_by_alpha, landmark_rotated_20)
			face_flipped = cv2.resize(face_flipped, (img_size,img_size))
			F_imgs.append(face_flipped.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_flipped_20.reshape(num_landmark))
			#print landmark_flipped_20

			# rotate the face
			face_rotated_by_alpha, landmark_rotated_25 = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), 25)
			landmark_rotated_25 = bbox.projectLandmark(landmark_rotated_25)
			face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))
			F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_rotated_25.reshape(num_landmark))
			#print landmark_rotated_25

			# flip with the rotation
			face_flipped, landmark_flipped_25 = flip(face_rotated_by_alpha, landmark_rotated_25)
			face_flipped = cv2.resize(face_flipped, (img_size,img_size))
			F_imgs.append(face_flipped.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_flipped_25.reshape(num_landmark))
			#print landmark_flipped_25

			# rotate the face
			face_rotated_by_alpha, landmark_rotated_30 = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), 30)
			landmark_rotated_30 = bbox.projectLandmark(landmark_rotated_30)
			face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))
			F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_rotated_30.reshape(num_landmark))
			#print landmark_rotated_30

			# flip with the rotation
			face_flipped, landmark_flipped_30 = flip(face_rotated_by_alpha, landmark_rotated_30)
			face_flipped = cv2.resize(face_flipped, (img_size,img_size))
			F_imgs.append(face_flipped.reshape((1, img_size,img_size)))
			F_landmarks.append(landmark_flipped_30.reshape(num_landmark))
			#print landmark_flipped_30

		face = cv2.resize(face, (img_size,img_size)).reshape(1,img_size,img_size)
		landmarkGt = landmarkGt.reshape(num_landmark)
		F_imgs.append(face)
		F_landmarks.append(landmarkGt)

	F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
	F_imgs = processImage(F_imgs)
	F_imgs, F_landmarks = shuffle(F_imgs, F_landmarks, random_state=42)
	
	with h5py.File(h5_path, 'w') as f:
		f['data'] = F_imgs.astype(np.float32)
		f['landmarks'] = F_landmarks.astype(np.float32)
	with open(txt_path, 'w') as f:
		f.write(h5_path)
		f.write(str(len(F_landmarks)))

if __name__ == '__main__':
	config = ConfigObj('config_300w.txt')
	train_txt = config['train_txt']
	valid_txt = config['valid_txt']
	h5_dir = 'h5_300w_15_alexnet'
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
