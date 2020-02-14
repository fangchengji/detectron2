import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec

__all__ = ["DenseboxNoBrranchConv32T"]


@BACKBONE_REGISTRY.register()
class DenseboxNoBrranchConv32T(Backbone):
    """Densebox_NoBrranchConv_32T"""
    def __init__(self, cfg, input_shape):
        """
        Arguments:
            28 conv layers
            very few channels
        """
        super().__init__()
        self._out_channels = cfg.MODEL.DENSEBOX.OUT_CHANNELS
        self._out_features = cfg.MODEL.DENSEBOX.OUT_FEATURES

        self.conv1 = BaseBlockRelu(3, 64, 3, stride=2, padding=1, bias=False)
        self.conv2_1 = BaseBlockRelu(64, 32, 3, stride=2, padding=1, bias=False)
        self.conv2_2 = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)
        self.conv2_3 = BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)

        # self.res2a_b1 = BaseBlock(32, 32, 1, stride=1, padding=0, bias=False)
        self.res2a_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2a_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.res2b_b2a = BaseBlockRelu(32, 32, 3, stride=1, padding=1, bias=False)
        self.res2b_b2b = BaseBlock(32, 32, 3, stride=1, padding=1, bias=False)

        self.conv3_1 = BaseBlockRelu(32, 64, 3, stride=2, padding=1, bias=False)
        self.conv3_2 = BaseBlockRelu(64, 32, 3, stride=1, padding=1, bias=False)
        self.conv3_3 = BaseBlockRelu(32, 64, 3, stride=1, padding=1, bias=False)

        # self.res3a_b1 = BaseBlock(64, 64, 1, stride=1, bias=False)
        self.res3a_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3a_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3b_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3b_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.res3c_b2a = BaseBlockRelu(64, 64, 3, stride=1, padding=1, bias=False)
        self.res3c_b2b = BaseBlock(64, 64, 3, stride=1, padding=1, bias=False)

        self.conv4_1 = BaseBlockRelu(64, 128, 3, stride=2, padding=1, bias=False)
        self.conv4_2 = BaseBlockRelu(128, 64, 3, stride=1, padding=1, bias=False)
        self.conv4_3 = BaseBlockRelu(64, 128, 3, stride=1, padding=1, bias=False)

        # self.res4a_b1 = BaseBlock(128, 128, 1, stride=1, padding=0, bias=False)
        self.res4a_b2a = BaseBlockRelu(128, 128, 3, stride=1, padding=1, bias=False)
        self.res4a_b2b = BaseBlock(128, 128, 3, stride=1, padding=1, bias=False)

        self.res4b_b2a = BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4b_b2b = BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4c_b2a = BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4c_b2b = BaseBlock(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)

        self.res4d_b2a = BaseBlockRelu(128, 128, (3, 5), stride=1, padding=(1, 2), bias=False)
        self.res4d_b2b = BaseBlock(128, self._out_channels, (3, 5), stride=1, padding=(1, 2), bias=False)

        self._out_feature_channels = {'res4d': self._out_channels}
        self._out_feature_strides = {'res4d': 16}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        stages = {}
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = x + self.res2a_b2b(self.res2a_b2a(x))
        x = F.relu_(x)

        x1 = self.res2b_b2a(x)
        x2 = self.res2b_b2b(x1)
        x = F.relu_(x2 + x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x1 = x
        x2 = self.res3a_b2b(self.res3a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res3b_b2b(self.res3b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res3c_b2b(self.res3c_b2a(x))
        x = F.relu_(x + x1)

        res3c = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x1 = x  # self.res4a_b1(x)
        x2 = self.res4a_b2b(self.res4a_b2a(x))
        x = F.relu_(x1 + x2)

        x1 = self.res4b_b2b(self.res4b_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4c_b2b(self.res4c_b2a(x))
        x = F.relu_(x + x1)

        x1 = self.res4d_b2b(self.res4d_b2a(x))
        x = F.relu_(x + x1)

        res4d = x
        stages['res4d'] = res4d

        return {k: stages[k] for k in self._out_features}


class BaseBlock(nn.Module):
    """BaseBlock"""
    def __init__(self, *args, **kargs):
        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        return x


class BaseBlockRelu(nn.Module):
    """BaseBlockRelu"""
    def __init__(self, *args, **kargs):
        super(BaseBlockRelu, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


class ConvBlock(nn.Module):
    """ConvBlock"""
    def __init__(self, *args, **kargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(*args, **kargs)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv(x)
        return x


class BaseBlockDeconvRelu(nn.Module):
    """BaseBlockDeconvRelu"""
    def __init__(self, *args, **kargs):
        super(BaseBlockDeconvRelu, self).__init__()
        self.conv = nn.ConvTranspose2d(*args, **kargs)
        self.bn = nn.BatchNorm2d(args[1])

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x


class SELayer(nn.Module):
    """SELayer"""
    def __init__(self, channel, reduction=16):
        """
        Squeeze and excitation
        Reference: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
        :param channel:
        :param reduction:
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

