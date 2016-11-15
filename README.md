This is the implementation of face landmark detection on 300-W dataset using caffe. It is based on the following projects:

https://github.com/luoyetx/deep-landmark (original implementation)

https://github.com/pminmin/caffe_landmark

https://github.com/ishay2b/VanillaCNN 

The success of landmark detection mainly relies on two aspects: (a) Data Augmentation and (B) Network.

For data augmentation, I use both rotation and bounding box perturbation. After data augmentation, there is a total of 30,301 samples and 5,878 samples for training and validation sets, respectively.

For network, I choose Vanilla CNN as the building block. The input size is 40*40 and the landmark positions has been scaled to [0,1]. 



# For prediction

python predict_vanilla_fd_one.py Model_68Point/_iter_1400000.caffemodel 314.jpg


# Evaluation Results

Inline-style: 
![alt text](https://github.com/cunjian/face_alignment/blob/master/demo_result.jpg "Logo Title Text 1")


