#!/usr/bin/env python3
# @Time    : 27/5/20 4:52 PM
# @Author  : fangcheng.ji
# @FileName: classification_head.py

import torch
import torch.nn as nn

from typing import List

from detectron2.layers import ShapeSpec


class ClassificationHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        in_channels = 0
        for shape in input_shape:
            in_channels += shape.channels
        self.num_classes = cfg.MODEL.CLASSIFICATION.NUM_CLASSES
        self.avtivation = cfg.MODEL.CLASSIFICATION.ACTIVATION

        # resnet fc for classification tasks
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.avtivation == "softmax":
            self.linear = nn.Linear(in_channels, self.num_classes + 1)
        else:
            self.linear = nn.Linear(in_channels, self.num_classes)

        # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "The 1000-way fully-connected layer is initialized by
        # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
        """
        xs = []
        for feature in features:
            x = self.avgpool(feature)
            x = torch.flatten(x, 1)
            xs.append(x)

        ff = torch.cat(xs, dim=1)
        logits = self.linear(ff)

        return logits