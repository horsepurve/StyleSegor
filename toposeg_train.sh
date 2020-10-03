# First finetuning COCO dataset pretrained model on augmented set

# If you would like to train from scratch on COCO, please see deeplab_resnet101_coco.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# --dataset pascal_aug \
# --model-zoo deeplab_resnet101_coco --aux \
# --lr 0.001 \
# --syncbn --ngpus 4 \
# --checkname res101

cp ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/train.txt ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/trainval.txt
# cat ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/val.txt >> ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/trainval.txt
echo "train samples:"
wc -l ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/train.txt
echo "val samples:"
wc -l ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/val.txt

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 CUDA_VISIBLE_DEVICES=0,2 python train.py \
--epochs 20 \
--lr 0.002 \
--syncbn --ngpus 2 \
--checkname HVSMR \
--workers 8 \
--batch-size 8 \
--model deeplab \
--backbone resnet101 --save_name 14378 \
--model-zoo deeplab_resnet101_coco --aux \
--resume res50_backup.params
# --eval
# --no-val
# --aux \
# --base-size 480 \
# --crop-size 480 \

# Finetuning on original set

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --model deeplab --aux --backbone resnet101 --lr 0.0001 --syncbn --ngpus 4 --checkname res101 --resume runs/pascal_aug/deeplab/res101/checkpoint.params
