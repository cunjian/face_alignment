### Introduction

This is the implementation of face landmark detection on 300-W dataset using Caffe. 

It performs the face alignment with 68-point annotation, which is more challenging than 5-point landmark annotation. It is purely trained on the 300W dataset, without using any external dataset. For face detection, DLIB library is used. Please install Caffe and DLIB ahead before playing with this model. 

For data augmentation, both rotation and bounding box perturbation are added. Samples outside of the bounaries have been removed. The current implementation supporst HDF5, which requires the loaded h5 file to be less than 2G. It can be splitted into several smaller files if the size exceeds the limit. 

For network, I choose Vanilla CNN as the building block. The input size is 40*40 and the landmark positions has been scaled to [0,1]. You can replace the backbone network with more advanced network structure. 

### Training:

1. Download the 300W dataset and run create_raw_300W_equal_bbox_train.py to obtain the annotation.

2. Prepare the dataset using generate_h5_300w_scale_V4.py; This performs data augmentation and generates the input for the Caffe. 

3. Run train.sh

### Prediction

python predict_vanilla_fd_one.py Model_68Point/_iter_1400000.caffemodel 314.jpg


### Evaluation Results


![alt text](https://github.com/cunjian/face_alignment/blob/master/demo_result.jpg "Logo Title Text 1")

![alt text](https://github.com/cunjian/face_alignment/blob/master/demo_result_19.jpg "Logo Title Text 1")

Images are either taken from the face landmark evaluation dataset or from the Internet. Copyright belongs to the owners.

The implementation is inspired from the following projects. 

References:

1. https://github.com/luoyetx/deep-landmark

2. https://github.com/ishay2b/VanillaCNN 
