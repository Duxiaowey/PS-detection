# Image manipulation detection
Paper: CVPR2018, [Learning Rich Features for Image Manipulation Detection](https://arxiv.org/pdf/1805.04953.pdf)  
Code based on [Faster-RCNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)  


# Environment
Python 3.6    
pytorch  1.3.1


# Pre-trained model
resnet101_caffe.pth


# Dataset
[Casia 1.0](http://forensics.idealtest.org)   
The name of the files in Sp may be something wrong. Use code in faster-rcnn/data/VOC2013 to modify the name and check it.   
Thanks to @namtpham to offer groudtruth for CASIA 1.0 and CASIA 2.0.    
The principle to rename files in origin dataset is to match them to GT.
Because the input of faster-rcnn is in the formation of VOC2007 and VOV2012, so I change the Dataset to this formation.

# Updating……
