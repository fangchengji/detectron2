#!/usr/bin/env python3
# @Time    : 18/5/20 11:41 AM
# @Author  : fangcheng.ji
# @FileName: pafpn.py

import torch.nn.functional as F
import math

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .fpn import FPN, LastLevelMaxPool, LastLevelP6P7, LastLevelP6
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone


class PAFPN(FPN):
    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        super().__init__(bottom_up, in_features, out_channels, norm, top_block, fuse_type)

        self.norm = norm
        up_convs = []
        fuse_convs = []

        use_bias = norm == ""
        for idx in range(len(self._out_features) - 1):
            up_norm = get_norm(norm, out_channels)
            fuse_norm = get_norm(norm, out_channels)

            up_conv = Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=up_norm
            )
            fuse_conv = Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=fuse_norm
            )
            weight_init.c2_xavier_fill(up_conv)
            weight_init.c2_xavier_fill(fuse_conv)
            stage = int(math.log2(self._out_feature_strides[self._out_features[idx]]))
            self.add_module("pafpn_up{}".format(stage), up_conv)
            self.add_module("pafpn_fuse{}".format(stage), fuse_conv)

            up_convs.append(up_conv)
            fuse_convs.append(fuse_conv)

        self.up_convs = up_convs
        self.fuse_convs = fuse_convs

        self._size_divisibility = max([v for _, v in self._out_feature_strides.items()])

    def forward(self, x):
        # fpn forward
        x = super().forward(x)
        # pa fpn
        x = [x[f] for f in self._out_features]
        results = [x[0]]
        for idx, (up_conv, fuse_conv) in enumerate(zip(self.up_convs, self.fuse_convs)):
            up_feature = up_conv(x[idx])
            if self.norm is not "":
                up_feature = F.relu_(up_feature)
            fuse_feature = fuse_conv(up_feature + x[idx + 1])
            if self.norm is not "":
                fuse_feature = F.relu_(fuse_feature)
            results.append(fuse_feature)
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))


@BACKBONE_REGISTRY.register()
def build_resnet_pafpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = PAFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_pafpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = PAFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone