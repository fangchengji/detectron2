#!/bin/bash

# to ensure code can find FashionNet module
export PYTHONPATH=$PYTHONPATH:/data/fangcheng.ji/detectron2/projects/FashionNet

cd projects/FashionNet
echo "The work directory should be detectron2/projects/FashionNet!!!!"
echo "Now it is at `pwd`"

CUDA_VISIBLE_DEVICES=3 python train_net.py --num-gpus 1 \
  --config-file configs/fashionnet_R_50.yaml \
  --dist-url auto \
  --resume
