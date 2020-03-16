# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_fashionnet_config(cfg):
    """
    Add config for fashionnet head.
    """
    _C = cfg

    # onnx export model input size, w, h
    _C.INPUT.SIZE = (800, 800)

    # Add sampler meta module
    _C.DATALOADER.SAMPLER_META = "fashionnet.samplers"

    # View the test error result
    _C.TEST.VIEW_ERROR = False

    # ---------------------------------------------------------------------------- #
    # FashionNet
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FASHIONNET = CN()
    # if export onnx model, it should only run forward the network
    # skip the preprocess and postprocess
    _C.MODEL.FASHIONNET.EXPORT_ONNX = False

    # ---------------------------------------------------------------------------- #
    # FashionNet Classification Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD = CN()
    # This is the classification tasks name.
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.TASK_NAMES = ["category", "part", "toward"]
    # This is the number of foreground classes for each task.
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CLASSES = [5, 3, 2]
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.ACTIVATION = 'softmax'
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.SCORE_THRESH = 0.5

    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CONVS = 4

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    _C.MODEL.FASHIONNET.CLASSIFICATION_HEAD.PRIOR_PROB = 0.01

    # ---------------------------------------------------------------------------- #
    # DenseBox
    # ---------------------------------------------------------------------------- #
    _C.MODEL.DENSEBOX = CN()
    _C.MODEL.DENSEBOX.OUT_CHANNELS = 128
    _C.MODEL.DENSEBOX.OUT_FEATURES = ['res4d']

    # ---------------------------------------------------------------------------- #
    # EfficientNet
    # ---------------------------------------------------------------------------- #
    _C.MODEL.EFFICIENTNET = CN()
    # From 0 to 7
    _C.MODEL.EFFICIENTNET.MODEL_SIZE = 0
    _C.MODEL.EFFICIENTNET.OUT_FEATURES = ['head']
