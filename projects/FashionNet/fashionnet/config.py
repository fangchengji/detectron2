# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_fashionnet_config(cfg):
    """
    Add config for fashionnet head.
    """
    _C = cfg

    # Add sampler meta module
    _C.DATALOADER.SAMPLER_META = "projects.FashionNet.fashionnet.samplers"

    # ---------------------------------------------------------------------------- #
    # FashionNet
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FASHIONNET = CN()

    # ---------------------------------------------------------------------------- #
    # FashionNet Classification Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD = CN()
    # This is the classification tasks name.
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.TASK_NAMES = ["category2_id", "part", "toward"]
    # This is the number of foreground classes for each task.
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CLASSES = [4, 3, 2]

    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CONVS = 4

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.PRIOR_PROB = 0.01

