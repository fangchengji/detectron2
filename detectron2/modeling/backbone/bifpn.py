#!/usr/bin/env python3
# @Time    : 8/5/20 3:43 PM
# @Author  : fangcheng.ji
# @FileName: bifpn.py

import torch
from torch import nn
from torch.nn import functional as F

import math

import fvcore.nn.weight_init as weight_init
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import get_norm, ShapeSpec

from .efficientnet import MemoryEfficientSwish, Swish, build_efficientnet_backbone
from .fpn import LastLevelMaxPool, LastLevelP6P7


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """
    def __init__(self, in_channels, out_channels=None, norm="", activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.
        use_bias = norm == ""
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False
        )
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias)

        self.norm = norm
        if self.norm is not "":
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            # self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
            self.bn = get_norm(norm, out_channels)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm is not "":
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPNBlock(nn.Module):
    def __init__(
        self, in_features, out_channels, norm="BN", epsilon=1e-4, onnx_export=False, attention=True
    ):
        """
        Args:
            out_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPNBlock, self).__init__()

        self.epsilon = epsilon
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Conv layers
        self.up_convs = []
        self.down_convs = []
        self.down_weights = []
        self.up_weights = []
        for idx, f in enumerate(in_features):
            if idx != len(in_features) - 1:
                conv = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
                self.add_module(f"down_conv_{f}", conv)
                self.down_convs.append(conv)

                # if no attention, just set weight to constant 1.
                weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=attention)
                # self.add_module(f"down_w_{f}", weight)
                self.down_weights.append(weight)

            if idx != 0:
                conv = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
                self.add_module(f"up_conv_{f}", conv)
                self.up_convs.append(conv)

                fuse_num = 3 if idx + 1 != len(in_features) else 2
                weight = nn.Parameter(torch.ones(fuse_num, dtype=torch.float32), requires_grad=attention)
                # self.add_module(f"up_w_{f}", weight)
                self.up_weights.append(weight)
        # add fuse weights to graph
        self.fuse_weights = nn.ParameterList([*self.down_weights, *self.up_weights])
        # init
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                weight_init.c2_msra_fill(module)

        # down path should reverse the order
        self.down_convs = self.down_convs[::-1]
        self.down_weights = self.down_weights[::-1]

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        # down path
        inputs = inputs[::-1]  # reverse the input order
        features_td = []
        for idx, (conv, w) in enumerate(zip(self.down_convs, self.down_weights)):
            w_down = F.relu(w)
            w_down = w_down / (torch.sum(w_down, dim=0) + self.epsilon)
            features_td.append(
                conv(self.swish(w_down[0] * inputs[idx + 1] + w_down[1] * F.interpolate(inputs[idx], scale_factor=2)))
            )

        # up path
        inputs = inputs[::-1]
        features_td = features_td[::-1]
        outs = [features_td[0]]
        for idx, (conv, w) in enumerate(zip(self.up_convs, self.up_weights)):
            w_up = F.relu(w)
            w_up = w_up / (torch.sum(w_up, dim=0) + self.epsilon)
            fuse_feature = w_up[0] * inputs[idx + 1] \
                         + w_up[1] * F.max_pool2d(outs[idx], kernel_size=3, stride=2, padding=1)
            if idx + 1 != len(self.up_convs):
                fuse_feature += w_up[2] * features_td[idx + 1]
            outs.append(conv(self.swish(fuse_feature)))

        return outs


class BiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, compound_coef, out_channels=-1, norm="SyncBN", top_block=None, export_onnx=False,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        bifpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

        use_bias = norm == ""
        self.lateral_convs = []
        for idx, in_channel in enumerate(in_channels):
            conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channels, kernel_size=1, bias=use_bias),
                get_norm(norm, out_channels),
            )
            weight_init.c2_xavier_fill(conv[0])
            stage = int(math.log2(in_strides[idx]))
            self.add_module(f"bifpn_lateral{stage}", conv)
            self.lateral_convs.append(conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # TODO: rebuild bifpn in more compatible way, such as FPN
        self.top_block = top_block
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        # self._out_feature_strides['p6'] = 64
        # self._out_feature_strides['p7'] = 128

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = max([v for _, v in self._out_feature_strides.items()])

        # top down and bottom up bifpn tower
        self.bifpn_tower = nn.Sequential(
            *[BiFPNBlock(
                in_features=self._out_features,
                out_channels=out_channels,
                norm=norm,
                onnx_export=export_onnx,
                attention=True if compound_coef < 6 else False
            ) for _ in range(bifpn_cell_repeats[compound_coef])]
        )

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features]

        for idx, conv in enumerate(self.lateral_convs):
            x[idx] = conv(x[idx])

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = x[self._out_features.index(self.top_block.in_feature)]
            x.extend(self.top_block(top_block_in_feature))

        results = self.bifpn_tower(x)

        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_efficientnet_bifpn_backbone(cfg, input_shape):
    bottom_up = build_efficientnet_backbone(cfg, input_shape)

    compound_coef = cfg.MODEL.EFFICIENTNET.COMPOUND_COEFFICIENT
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS

    bifpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
    out_channels = out_channels if out_channels > 0 else bifpn_num_filters[compound_coef]
    in_channels_top = bottom_up.output_shape()['p5'].channels
    top_levels = cfg.MODEL.BIFPN.TOP_LEVELS
    export_onnx = cfg.MODEL.EXPORT_ONNX

    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    if top_levels == 1:
        top_block = LastLevelMaxPool(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        compound_coef=compound_coef,
        out_channels=out_channels,
        norm=cfg.MODEL.BIFPN.NORM,
        top_block=top_block,
        export_onnx=export_onnx,
    )

    return backbone
