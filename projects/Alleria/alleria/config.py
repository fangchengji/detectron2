#!/usr/bin/env python3
# @Time    : 4/6/20 5:47 PM
# @Author  : fangcheng.ji
# @FileName: config.py

# from detectron2.config import CfgNode as CN


def add_alleria_config(cfg):
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # Data Augmentation
    # ---------------------------------------------------------------------------- #
    # mosaic augmentation
    _C.DATALOADER.MOSAIC_PROB = 0.33
    # mix-up augmentation
    _C.DATALOADER.MIXUP_PROB = 0.34
    _C.DATALOADER.CUTOUT_PROB = 0.0
