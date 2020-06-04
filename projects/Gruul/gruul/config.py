#!/usr/bin/env python3
# @Time    : 1/6/20 6:26 PM
# @Author  : fangcheng.ji
# @FileName: config.py

from detectron2.config import CfgNode as CN


def add_gruul_config(cfg):
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # Classification Network
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CLASSIFICATION = CN()
    _C.MODEL.CLASSIFICATION.NUM_CLASSES = 80
    _C.MODEL.CLASSIFICATION.IN_FEATURES = ["linear"]
    _C.MODEL.CLASSIFICATION.SCORE_THRESH = 0.5
    _C.MODEL.CLASSIFICATION.ACTIVATION = 'sigmoid'

    _C.MODEL.CLASSIFICATION.FOCAL_LOSS_GAMMA = 2.0
    _C.MODEL.CLASSIFICATION.FOCAL_LOSS_ALPHA = 1.0