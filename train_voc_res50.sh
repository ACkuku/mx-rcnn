python train.py \
    --dataset voc \
    --network resnet50 \
    --pretrained ./model/resnet-50-0000.params \
    --save-prefix ./model/resnet50_sync_bn_model/resnet-50-sync-bn \
    --gpus 0,1
