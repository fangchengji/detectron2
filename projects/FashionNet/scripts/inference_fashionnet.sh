#!/bin/bash

# to ensure code can find FashionNet module
export PYTHONPATH=$PYTHONPATH:/data/fangcheng.ji/detectron2/projects/FashionNet

cd projects/FashionNet
echo "The work directory should be detectron2/projects/FashionNet!!!!"
echo "Now it is at `pwd`"

python demo.py --config-file configs/fashionnet_R_50.yaml \
  --output output/9w \
  --input /data/fangcheng.ji/datasets/deepfashion2/train_test/image \
  --opts MODEL.WEIGHTS output/model_final.pth
