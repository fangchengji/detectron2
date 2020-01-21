#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python projects/FashionNet/train_net.py --num-gpus 2 --config-file projects/FashionNet/configs/fashionnet_R_50.yaml
