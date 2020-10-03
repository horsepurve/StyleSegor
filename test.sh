# python test.py --dataset ade20k --model-zoo fcn_resnet50_ade --eval

grep _v2_ val.txt > ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/test.txt

# cat ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/val.txt > ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/test.txt
head ~/.mxnet/datasets/voc/VOC2012/ImageSets/Segmentation/test.txt

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 CUDA_VISIBLE_DEVICES=3 python test.py \
--dataset pascal_voc --aux \
--checkname HVSMR \
--ngpus 1 \
--model deeplab \
--backbone resnet101 \
--resume /home/chunweim/chunweim/projects/SemiSegor/run/runs/pascal_voc/deeplab/HVSMR/res50-v2.params \
--outdir outdir_no_style_v2_1scale --scoredir scoredir_no_style_v2_1scale
# --model-zoo deeplab_resnet152_coco \
# --eval 