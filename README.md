This is the implementation of face landmark detection on 300-W dataset using caffe. It is based on the following projects:

https://github.com/luoyetx/deep-landmark (original implementation)

https://github.com/pminmin/caffe_landmark

https://github.com/ishay2b/VanillaCNN 

The success of landmark detection mainly relies on two aspects: (a) Data Augmentation and (B) Network.

For data augmentation, I use both rotation and bounding box perturbation. After data augmentation, there is a total of 30,301 samples and 5,878 samples for training and validation sets, respectively.

For network, I choose Vanilla CNN as the building block. The input size is 40*40 and the landmark positions has been scaled to [0,1]. 


# To generate dataset from 300W
cd dataset/300w

python create_raw_300W_equal_bbox_train.py

python create_raw_300W_equal_bbox_test.py

Use ratio=0, if you want to have a tight bounding box
Use ratio>0, if you want to enlarge the bounding box

# For prediction
cd evaluate

python predict_vanilla.py ../train_300w_68_vanilla/model/_iter_370000.caffemodel

# For training
cd train_300w_68_vanilla
bash train.sh

# Evaluation Results


