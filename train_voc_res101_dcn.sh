python train.py \
    --dataset voc \
    --network resnet101 \
    --use-deformable-conv \
    --pretrained ./model/resnet-101-0000.params \
    --save-prefix ./model/resnet101_dcn_model/resnet-101-dcn \
    --rcnn-batch-size 1 \
    --gpus 0,1
