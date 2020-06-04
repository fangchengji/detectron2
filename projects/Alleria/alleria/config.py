#!/usr/bin/env python3
# @Time    : 4/6/20 5:47 PM
# @Author  : fangcheng.ji
# @FileName: config.py

from detectron2.config import CfgNode as CN


def add_alleria_config(cfg):
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # Data Augmentation
    # ---------------------------------------------------------------------------- #
    _C.DATALOADER = CN()
    # mosaic augmentation
    _C.DATALOADER.MOSAIC_AUGMENTATION = 0.0
    # mix-up augmentation
    _C.DATALOADER.MIXUP_AUGMENTATION = 0.0
    # hsv augmentation
    _C.DATALOADER.HSV_AUGMENTATION = 0.0

