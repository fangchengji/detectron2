#!/usr/bin/env python3
# @Time    : 8/5/20 3:43 PM
# @Author  : fangcheng.ji
# @FileName: bifpn.py

import torch
from torch import nn
from torch.nn import functional as F

import math

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import get_norm, ShapeSpec

from .efficientnet import Conv2dStaticSamePadding, MemoryEfficientSwish, Swish, build_efficientnet_backbone


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

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

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
    """
    modified by Zylo117
    """

    def __init__(
        self, out_channels, in_channels, first_time=False, norm="BN", epsilon=1e-4, onnx_export=False, attention=True
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
        self.norm = get_norm(norm, out_channels)

        # Conv layers
        self.conv6_up = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(out_channels, norm=norm, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[2], out_channels, 1),
                # nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                self.norm,
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[1], out_channels, 1),
                # nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                self.norm,
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[0], out_channels, 1),
                # nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                self.norm,
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[2], out_channels, 1),
                # nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                self.norm,
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[1], out_channels, 1),
                # nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                self.norm,
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[2], out_channels, 1),
                # nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                self.norm,
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

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

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

        extra_h = h_cover_len - w
        extra_v = v_cover_len - h

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class BiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, compound_coef, out_channels=-1, norm="SyncBN", export_onnx=False,
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

        # backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        bifpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        bifpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

        out_channels = out_channels if out_channels > 0 else bifpn_num_filters[compound_coef]

        # conv_channel_coef = {
        #     # the channels of P3/P4/P5.
        #     0: [40, 112, 320],
        #     1: [40, 112, 320],
        #     2: [48, 120, 352],
        #     3: [48, 136, 384],
        #     4: [56, 160, 448],
        #     5: [64, 176, 512],
        #     6: [72, 200, 576],
        #     7: [72, 200, 576],
        # }

        self.bifpn = nn.Sequential(
            *[BiFPNBlock(
                out_channels=out_channels,
                in_channels=in_channels,
                first_time=True if _ == 0 else False,
                norm=norm,
                onnx_export=export_onnx,
                attention=True if compound_coef < 6 else False
            )
            for _ in range(bifpn_cell_repeats[compound_coef])]
        )

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # TODO: rebuild bifpn in more compatible way, such as FPN
        self._out_feature_strides['p6'] = 64
        self._out_feature_strides['p7'] = 128

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = max([v for _, v in self._out_feature_strides.items()])

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
        results = self.bifpn(x)

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
    # in_channels_top = bottom_up.output_shape()['p5'].channels
    export_onnx = cfg.MODEL.EXPORT_ONNX

    backbone = BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        compound_coef=compound_coef,
        out_channels=out_channels,
        norm=cfg.MODEL.BIFPN.NORM,
        export_onnx=export_onnx,
    )

    return backbone
