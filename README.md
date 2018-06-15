# Faster R-CNN in MXNet

![example detections](https://cloud.githubusercontent.com/assets/13162287/22101032/92085dc0-de6c-11e6-9228-67e72606ddbc.png)

Region Proposal Network solves object detection as a regression problem 
from the objectness perspective. Bounding boxes are predicted by applying 
learned bounding box deltas to base boxes, namely anchor boxes across 
different positions in feature maps. Training process directly learns a 
mapping from raw image intensities to bounding box transformation targets.

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

Faster R-CNN utilize an alternate optimization training process between RPN 
and Fast R-CNN. Fast R-CNN weights are used to initiate RPN for training.
The approximate joint training scheme does not backpropagate rcnn training
error to rpn training.

## Experiments

| Indicator | py-faster-rcnn (caffe resp.) | mx-rcnn (this reproduction) |
| :-------- | :--------------------------- | :-------------------------- |
| Training speed [1] | 2.5 img/s training, 5 img/s testing | 3.8 img/s in training, 12.5 img/s testing |
| Valset performance [2] | mAP 73.2               | mAP 75.97                   |
| Memory usage [3]  | 11G for Fast R-CNN     | 4.6G for Fast R-CNN         |
| Parallelization  [4] | None              | 3.8 img/s to 6 img/s for 2 GPUs |
| Extensibility [5] | Old framework and base networks | ResNet           |

[1] On Ubuntu 14.04.5 with device Titan X, cuDNN enabled.
    The experiment is VGG-16 end-to-end training.  
[2] VGG network. Trained end-to-end on VOC07trainval+12trainval, tested on VOC07 test.  
[3] VGG network. Fast R-CNN is the most memory expensive process.  
[4] VGG network (parallelization limited by bandwidth).
    ResNet-101 speeds up from 2 img/s to 3.5 img/s.  
[5] py-faster-rcnn does not support ResNet or recent caffe version.

| Method | Network | Training Data | Testing Data | Reference | Result | Link  |
| :----- | :------ | :------------ | :----------- | :-------: | :----: | :---: |
| Faster R-CNN end-to-end | VGG16 | VOC07 | VOC07test | 69.9 | 70.23 | [Dropbox](https://www.dropbox.com/s/gfxnf1qzzc0lzw2/vgg_voc07-0010.params?dl=0) |
| Faster R-CNN end-to-end | VGG16 | VOC07+12 | VOC07test | 73.2 | 75.97 | [Dropbox](https://www.dropbox.com/s/rvktx65s48cuyb9/vgg_voc0712-0010.params?dl=0) |
| Faster R-CNN end-to-end | ResNet-101 | VOC07+12 | VOC07test | 76.4 | 79.35 | [Dropbox](https://www.dropbox.com/s/ge2wl0tn47xezdf/resnet_voc0712-0010.params?dl=0) |
| Faster R-CNN end-to-end | VGG16 | COCO train | COCO val | 21.2 | 22.8 | [Dropbox](https://www.dropbox.com/s/e0ivvrc4pku3vj7/vgg_coco-0010.params?dl=0) |
| Faster R-CNN end-to-end | ResNet-101 | COCO train | COCO val | 27.2 | 26.1 | [Dropbox](https://www.dropbox.com/s/bfuy2uo1q1nwqjr/resnet_coco-0010.params?dl=0) |

The above experiments were conducted on [this version of repository](https://github.com/precedenceguo/mx-rcnn/tree/6a1ab0eec5035a10a1efb5fc8c9d6c54e101b4d0)
with [a modified MXNet based on 0.9.1 nnvm pre-release](https://github.com/precedenceguo/mxnet/tree/simple).

## Set up environment
* Install Python package `mxnet` or `mxnet-cu90`, `cython` and `opencv-python matplotlib pycocotools tqdm`.

## Download data and label
Follow `py-faster-rcnn` for data preparation instructions.
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) should be in `data/VOCdevkit` containing `VOC2007`, `VOC2012` and `annotations`.
* [MSCOCO](http://mscoco.org/dataset/) should be in `data/coco` containing `train2017`, `val2017` and `annotations/instances_train2017.json`, `annotations/instances_val2017.json`.

## Download pretrained ImageNet models
* [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) should be at `model/vgg16-0000.params` from [MXNet model zoo](http://data.dmlc.ml/models/imagenet/vgg/).
* [ResNet](https://github.com/tornadomeet/ResNet) should be at `model/resnet-101-0000.params` from [MXNet model zoo](http://data.dmlc.ml/models/imagenet/resnet/).

### History
* May 25, 2016: We released Fast R-CNN implementation.
* July 6, 2016: We released Faster R-CNN implementation.
* July 23, 2016: We updated to MXNet module solver.
* Oct 10, 2016: tornadomeet released approximate end-to-end training.
* Oct 30, 2016: We updated to MXNet module inference.
* Jan 19, 2017: We accelerated our pipeline and supported ResNet training.

### Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe),
[tornadomeet/mx-rcnn](https://github.com/tornadomeet/mx-rcnn),
[MS COCO API](https://github.com/pdollar/coco).  
Thanks to tornadomeet for end-to-end experiments and MXNet contributers for helpful discussions.

### References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. "The pascal visual object classes (voc) challenge." International journal of computer vision 88, no. 2 (2010): 303-338.
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
7. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
8. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
9. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. "Microsoft COCO: Common Objects in Context" In European Conference on Computer Vision, pp. 740-755. Springer International Publishing, 2014.
