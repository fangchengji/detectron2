#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`
echo $PYTHONPATH
CUDA_VISIBLE_DEVICES=2,3 python tools/train_net.py --num-gpus 2 --config-file configs/Fashion/fashionnet_R_50.yaml