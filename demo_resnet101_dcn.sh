python demo.py \
    --network resnet101 \
    --dataset voc \
    --gpu 0 \
    --vis \
    --vis-thresh 0.7 \
    --image ./data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
    --params ./model/resnet101_dcn_model/resnet-101-dcn-0010.params \
    --use-soft-nms False \
    --soft-nms-thresh 0.6 \
    --max-per-image 10