#!/usr/bin/env python3
# @Time    : 23/6/20 11:41 AM
# @Author  : fangcheng.ji
# @FileName: sepc.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from detectron2.layers import sepc_conv

from .build import BACKBONE_REGISTRY
from .fpn import FPN, LastLevelP6P7, LastLevelP6
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_resnet_backbone, build_resnest_backbone


class SEPC(FPN):
    """
    from this repo https://github.com/jshilong/SEPC
    """
    def __init__(self,
                 bottom_up,
                 in_features,
                 out_channels=256,
                 norm="",
                 top_block=None,
                 fuse_type="sum",
                 in_channels=[256] * 5,
                 num_outs=5,
                 pconv_deform=True,
                 lcconv_deform=False,
                 iBN=False,
                 Pconv_num=4,
                 ):
        super().__init__(bottom_up, in_features, out_channels, norm, top_block, fuse_type)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5

        self.iBN = iBN
        self.Pconvs = nn.ModuleList()

        for i in range(Pconv_num):
            self.Pconvs.append(PConvModule(in_channels[i], out_channels, iBN=self.iBN, part_deform=pconv_deform))

        # self.lconv = sepc_conv(256, 256, kernel_size=3, dilation=1, part_deform=lcconv_deform)
        # self.cconv = sepc_conv(256, 256, kernel_size=3, dilation=1, part_deform=lcconv_deform)
        # self.relu = nn.ReLU()
        # if self.iBN:
        #     self.lbn = nn.BatchNorm2d(256)
        #     self.cbn = nn.BatchNorm2d(256)
        # self.init_weights()

        # Backbone attribute
        self._size_divisibility = max([v for _, v in self._out_feature_strides.items()])

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for str in ["l", "c"]:
            m = getattr(self, str + "conv")
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    # @auto_fp16()
    def forward(self, x):
        # fpn forward
        x = super().forward(x)
        inputs = [x[f] for f in self._out_features]

        # spec code
        assert len(inputs) == len(self.in_channels)
        x = inputs
        for pconv in self.Pconvs:
            x = pconv(x)

        # cls = [self.cconv(level, item) for level, item in enumerate(x)]
        # loc = [self.lconv(level, item) for level, item in enumerate(x)]
        # if self.iBN:
        #     cls = iBN(cls, self.cbn)
        #     loc = iBN(loc, self.lbn)
        # outs = [[self.relu(s), self.relu(l)] for s, l in zip(cls, loc)]
        # return tuple(outs)

        results = x
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))


class PConvModule(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 kernel_size=[3, 3, 3],
                 dilation=[1, 1, 1],
                 groups=[1, 1, 1],
                 iBN=False,
                 part_deform=False,
                 ):
        super(PConvModule, self).__init__()

        #     assert not (bias and iBN)
        self.iBN = iBN
        self.Pconv = nn.ModuleList()
        self.Pconv.append(
            sepc_conv(in_channels, out_channels, kernel_size=kernel_size[0], dilation=dilation[0], groups=groups[0],
                      padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2, part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels, out_channels, kernel_size=kernel_size[1], dilation=dilation[1], groups=groups[1],
                      padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2, part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels, out_channels, kernel_size=kernel_size[2], dilation=dilation[2], groups=groups[2],
                      padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2, stride=2, part_deform=part_deform))

        if self.iBN:
            self.bn = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.Pconv:
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):

            temp_fea = self.Pconv[1](level, feature)
            if level > 0:
                temp_fea += self.Pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.interpolate(self.Pconv[0](level, x[level + 1]),
                                          size=[temp_fea.size(2), temp_fea.size(3)],
                                          mode='bilinear', align_corners=False)
            next_x.append(temp_fea)
        if self.iBN:
            next_x = iBN(next_x, self.bn)
        next_x = [self.relu(item) for item in next_x]
        return next_x


def iBN(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_sepc_backbone(cfg, input_shape: ShapeSpec):
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
    backbone = SEPC(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone


@BACKBONE_REGISTRY.register()
def build_fcos_resnest_sepc_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnest_backbone(cfg, input_shape)
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
    backbone = SEPC(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone