# coding: utf-8
# enlarge the bounding box by a ratio of 0.2
# use rotation [5,30] and [-5,-30]
# flip only works for 5 point
# use bounding box perturbation
import numpy as np
import h5py
import sys
import os
import cv2
import pdb
from os.path import exists, join, isfile
from configobj import ConfigObj
from sklearn.utils import shuffle
from common.utils import processImage, flip, BBox, check_bbox, rotate, drawLandmark
from common.get_data import load_celeba_data, get_train_val_test_list, getDataFromTXT, getDataFromTXT_68,getDataFromTXT_68_scale, getDataFromTXT_15, getDataFromTXT_5,getDataFromTXT_19

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
	img_size=40
	num_landmark=68*2
        num_sample=0
	for (imgpath, bbox, landmarkGt) in data:
		img = cv2.imread(imgpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		print ("processing %s") % imgpath
		if not check_bbox(img, bbox): 
			print 'BBox out of range.'
			continue
		face = img[bbox.top:bbox.bottom, bbox.left:bbox.right]

		if augment:
			# flip the face
			#face_flip, landmark_flip = flip(face, landmarkGt)
			#face_flip = cv2.resize(face_flip, (img_size,img_size)).reshape(1,img_size,img_size)

			#fit=0
			#for i in range(0,num_landmark/2):
			#    if landmark_flip[i,0]<0 or landmark_flip[i,0]>1 or landmark_flip[i,1]<0 or landmark_flip[i,1]>1:
                        #        fit=1
                        #        break
			#if fit==0:
			    #print landmark_flipped_alpha                               
			#    F_imgs.append(face_flip)
			#    F_landmarks.append(landmark_flip.reshape(num_landmark))

			#print landmark_flip
			#angles=[5,10,15,20,25,30,-5,-10,-15,-20,-25,-30]
			#for alpha in angles:
			    # rotate the face
			    #face_rotated_by_alpha, landmark_rotated_alpha = rotate(img, bbox,bbox.reprojectLandmark(landmarkGt), alpha)
			    #landmark_rotated_alpha = bbox.projectLandmark(landmark_rotated_alpha)
			    #face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))

			    #fit=0
			    #for i in range(0,num_landmark/2):
			    #    if landmark_rotated_alpha[i,0]<0 or landmark_rotated_alpha[i,0]>1 or landmark_rotated_alpha[i,1]<0 or landmark_rotated_alpha[i,1]>1:
			    #        fit=1
			    #        break
			    #if fit==0:
                                #print landmark_rotated_alpha		    
			    #    F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
                            #    F_landmarks.append(landmark_rotated_alpha.reshape(num_landmark))
			    


			        #print landmark_rotated_5
			        # flip with the rotation
			        #face_flipped_alpha, landmark_flipped_alpha = flip(face_rotated_by_alpha, landmark_rotated_alpha)
			        #face_flipped_alpha = cv2.resize(face_flipped_alpha, (img_size,img_size))
			        #fit=0
			        #for i in range(0,num_landmark/2):
			        #    if landmark_flipped_alpha[i,0]<0 or landmark_flipped_alpha[i,0]>1 or landmark_flipped_alpha[i,1]<0 or landmark_flipped_alpha[i,1]>1:
			        #        fit=1
			        #        break
			        #if fit==0:
			            #print landmark_flipped_alpha                               
			        #    F_imgs.append(face_flipped_alpha.reshape((1, img_size,img_size)))
                                #    F_landmarks.append(landmark_flipped_alpha.reshape(num_landmark))

                                # debug
			        #center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
			        #rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
			        #img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
			        #landmark_rotated_alpha = bbox.reprojectLandmark(landmark_rotated_alpha) # will affect the flip "landmark_rotated_alpha"
			        #img_rotated_by_alpha = drawLandmark(img_rotated_by_alpha, bbox, landmark_rotated_alpha)
			        #fp = 'debug_results/'+ str(num_sample)+'.jpg'
			        #cv2.imwrite(fp, img_rotated_by_alpha)
			        #num_sample=num_sample+1
                                
			        # debug

			#use bounding box perturbation
			y=bbox.top
			x=bbox.left
			h=bbox.bottom-bbox.top
			w=bbox.right-bbox.left
			# original landmark position
			landmarkGT_scale=bbox.reprojectLandmark(landmarkGt)
			for cur_scale in [0.83, 0.91, 1.0, 1.10, 1.21]:
			    for cur_x in [-0.17,0, 0.17]:
			        for cur_y in [-0.17,0, 0.17]:
			            s_n=1/cur_scale
			            x_n=-cur_x/cur_scale
			            y_n=-cur_y/cur_scale

			            x_temp=int(x-(x_n*w/s_n))
			            y_temp=int(y-(y_n*h/s_n))
			            w_temp=int(w/s_n)
			            h_temp=int(h/s_n)
			            # generate new bounding box	
			            bbox_left=x_temp
			            bbox_right=x_temp+w_temp
			            bbox_top=y_temp
			            bbox_bottom=y_temp+h_temp			
			            new_bbox = map(int, [bbox_left, bbox_right, bbox_top, bbox_bottom])
			            new_bbox = BBox(new_bbox)
			            if not check_bbox(img, new_bbox): 
			                print 'BBox out of range.'
			                continue

			            # project landmark onto the new bounding box
			            new_landmarkGT=new_bbox.projectLandmark(landmarkGT_scale)
                                    new_landmarkGT_org=new_landmarkGT.copy()

			            angles=[5,10,15,20,25,30,-5,-10,-15,-20,-25,-30]
			            for alpha in angles:
			                # rotate the face
			                face_rotated_by_alpha, landmark_rotated_alpha = rotate(img, new_bbox,new_bbox.reprojectLandmark(new_landmarkGT), alpha)
			                landmark_rotated_alpha = new_bbox.projectLandmark(landmark_rotated_alpha)
			                face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size,img_size))

			                fit=0
			                for i in range(0,num_landmark/2):
			                    if landmark_rotated_alpha[i,0]<0 or landmark_rotated_alpha[i,0]>1 or landmark_rotated_alpha[i,1]<0 or landmark_rotated_alpha[i,1]>1:
			                        fit=1
			                        break
			                if fit==0:
			                    #print landmark_rotated_alpha		    
			                    F_imgs.append(face_rotated_by_alpha.reshape((1, img_size,img_size)))
			                    F_landmarks.append(landmark_rotated_alpha.reshape(num_landmark))


			                    # debug
			                    #center = ((new_bbox.left+new_bbox.right)/2, (new_bbox.top+new_bbox.bottom)/2)
			                    #rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
			                    #img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
			                    #landmark_rotated_alpha = new_bbox.reprojectLandmark(landmark_rotated_alpha) # will affect the flip "landmark_rotated_alpha"
			                    #img_rotated_by_alpha = drawLandmark(img_rotated_by_alpha, new_bbox, landmark_rotated_alpha)
			                    #fp = 'debug_results/'+ str(num_sample)+'.jpg'
			                    #cv2.imwrite(fp, img_rotated_by_alpha)
			                    #num_sample=num_sample+1
			                    # debug

			            # project landmark onto the new bounding box
			            landmarkGT_project=new_landmarkGT_org.copy() # error is fixed here
			            #print landmarkGT_project

			            fit=0
			            for i in range(0,num_landmark/2):
			                if landmarkGT_project[i,0]<0 or landmarkGT_project[i,0]>1 or landmarkGT_project[i,1]<0 or landmarkGT_project[i,1]>1:
			                    fit=1
			                    break
			            if fit==0:
			                #print landmarkGT_project
			                #if landmarkGT_project[i,0]<0 or landmarkGT_project[i,0]>1 or landmarkGT_project[i,1]<0 or landmarkGT_project[i,1]>1:
			                #    pdb.set_trace()		    
			                cropped_face=img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
			                cropped_face = cv2.resize(cropped_face, (img_size,img_size)).reshape(1,img_size,img_size)
                                        F_imgs.append(cropped_face)
		                        F_landmarks.append(landmarkGT_project.reshape(num_landmark))	            
			            
		                        # debug on the bounding box perturbation
		                        #landmark_org = new_bbox.reprojectLandmark(landmarkGT_project) # will affect the flip "landmark_rotated_alpha"
		                        #img_debug = drawLandmark(img, new_bbox, landmark_org)
		                        #fp = 'debug_results/'+ str(num_sample)+'_bbox.jpg'
		                        #cv2.imwrite(fp, img_debug)
		                        #num_sample=num_sample+1			           
			            


		face = cv2.resize(face, (img_size,img_size)).reshape(1,img_size,img_size)

		fit=0
		for i in range(0,num_landmark/2):
		    if landmarkGt[i,0]<0 or landmarkGt[i,0]>1 or landmarkGt[i,1]<0 or landmarkGt[i,1]>1:
		        fit=1
		        break
		if fit==0:
		    #print landmark_flipped_alpha                               
		    F_imgs.append(face)
		    F_landmarks.append(landmarkGt.reshape(num_landmark))

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
	h5_dir = 'h5_300w_68_40_box'
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
		train_data = getDataFromTXT_68(train_txt)
		val_data = getDataFromTXT_68(valid_txt)

		generate_h5py(train_data, join(h5_dir,'train.h5').replace('\\','/'), join(h5_dir,'train.txt'), augment=True)
		generate_h5py(val_data, join(h5_dir,'valid.h5').replace('\\','/'), join(h5_dir,'valid.txt'), augment=True)
