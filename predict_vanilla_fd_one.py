# coding: utf-8
# 2016/11/9
# Use face detection, other than the provided bounding box
import os
import cv2
import dlib
import numpy as np
import caffe
from configobj import ConfigObj 
import sys
sys.path.append("..")
from common.utils import BBox, drawLandmark,drawLandmark_multiple, check_bbox, processImage,scale
from common.get_data import getDataFromTXT_68_scale,getDataFromTXT_15_scale, load_celeba_data, get_train_val_test_list

class Predict(object):
	def __init__(self, net, model):
		self.net = net
		self.model = model
		self.cnn = caffe.Net(str(net), str(model), caffe.TEST)

	def forward(self, face, layer='Dense2'):
		'''
		face has been preprocessed to (1,C,H,W) before feeding to cnn.
		'''
		out = self.cnn.forward_all(data=np.asarray([face]))
		landmark = out[layer][0]
		landmark = landmark.reshape(68,2)

		return landmark

def load_test_img(gray, bbox):
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face = gray[bbox.top:bbox.bottom, bbox.left:bbox.right]
	face = cv2.resize(face, (40,40)).reshape(1,1,40,40)
	face = processImage(np.asarray(face))

	return face

if __name__ == '__main__':


	net = Predict('../train_300w_68_vanilla/deploy.prototxt', sys.argv[1])

        # cascade file
        hc = cv2.CascadeClassifier("haarcascades/xml/haarcascade_frontalface_alt2.xml")
        ratio = 0.1
	# Test for original dataset 
        img_path=sys.argv[2]
        img = cv2.imread(img_path)    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
        # perform face detection using OpenCV
        #faces=hc.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        # perform face detection using dlib
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)
        if len(faces)==0:
            print 'NO face is detected!'
        #for face in faces:
        for k, face in enumerate(faces):
            #cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255,     0, 0), 3) # (x,y,w,h)
            # used by opencv
            #x_min=face[0]
            #y_min=face[1]
            #x_max=face[0] + face[2]
            #y_max=face[1]+face[3]
            # used by dlib
            x_min=face.left()
            y_min=face.top()
            x_max=face.right()
            y_max=face.bottom()
            w, h = x_max-x_min, y_max-y_min
            w = h = min(w, h)
            x_new = x_min - w*ratio
            y_new = y_min - h*ratio
            w_new = w*(1 + 2*ratio)
            h_new = h*(1 + 2*ratio)
            new_bbox = map(int, [x_new, x_new+w_new, y_new, y_new+h_new])
            new_bbox = BBox(new_bbox)
            #print bbox_left,bbox_top,bbox_right,bbox_bottom
            if not check_bbox(gray.transpose(), new_bbox): 
                print 'BBox out of range.'
                continue
            face = load_test_img(gray, new_bbox)
            landmark = net.forward(face, 'Dense2')
            #print landmark
            landmark = new_bbox.reprojectLandmark(landmark)
            img = drawLandmark_multiple(img, new_bbox, landmark)

            cv2.imwrite('demo_result.jpg', img)





