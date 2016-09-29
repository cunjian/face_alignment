This is the implementation of face landmark detection on 300-W dataset using caffe. It is based on the following projects:

https://github.com/luoyetx/deep-landmark (original implementation)

https://github.com/pminmin/caffe_landmark

# To generate dataset from 300W
cd dataset/300w

python create_raw_300W_equal_bbox_train.py

python create_raw_300W_equal_bbox_test.py

# For prediction
cd evaluate

python predict.py ../train_300w_5/model/_iter_200000.caffemodel

# For training
cd train_300w_5

bash train.sh

I would plan to add training models for 68 point landmark.
