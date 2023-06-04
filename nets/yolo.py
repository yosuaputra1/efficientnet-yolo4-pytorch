from collections import OrderedDict

import torch
import torch.nn as nn

from nets.efficientnet import EfficientNet as EffNet
from .mobilenet_v3 import mobilenet_v3


class MobileNetV3(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:13](out3)
        out5 = self.model.features[13:16](out4)
        return out3, out4, out5


class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        out_feats = [feature_maps[2], feature_maps[3], feature_maps[4]]
        return out_feats


def cbl(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride = 1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


# ---------------------------------------------------#
# three convolution block
# conv2d (in_channels, out_channels, kernel_size, stride)
# ---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        cbl(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        cbl(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
# Five convolution block
# ---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        cbl(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        cbl(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        cbl(filters_list[1], filters_list[0], 1),
    )
    return m


# ---------------------------------------------------#
# SPP structure, using pooling kernels of different sizes for pooling
# stack after pooling
# ---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # nn.MaxPool2d(kernel_size, stride, padding)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


# ---------------------------------------------------#
# convolution + upsampling
# ---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            cbl(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


# ---------------------------------------------------#
# Finally get the output of yolov4
# ---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, load_weights=False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#
        #   生成efficientnet的主干模型，以efficientnetB0为例
        #   获得三个有效特征层，他们的shape分别是：
        #   52, 52, 40
        #   26, 26, 112
        #   13, 13, 320
        #---------------------------------------------------#
        self.backbone = EfficientNet(phi, load_weights=load_weights)

        out_filters = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [80, 224, 640],
        }[phi]

        self.conv1 = make_three_conv([512, 1024], out_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = cbl(out_filters[1], 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = cbl(out_filters[0], 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        final_out_filter2 = len(anchors_mask[0]) * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2], 128)

        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)
        final_out_filter1 = len(anchors_mask[1]) * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1], 256)

        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)
        final_out_filter0 = len(anchors_mask[2]) * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0], 512)

    def forward(self, x):
        # --------------------------------------------------- #
        # Obtain three effective feature layers, their shapes are:
        # 52, 52, 40
        # 26, 26, 112
        # 13, 13, 320
        # --------------------------------------------------- #
        x2, x1, x0 = self.backbone(x)

        # SPP:
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], axis=1)
        P5 = self.make_five_conv4(P5)

        # P3: 52, 52, 128 =ConvBlock=> P3: 52, 52, 256 =1*1conv2d=> P3: 52, 52, num_anchors * (5 + num_classes)
        out2 = self.yolo_head3(P3)
        # P4: 26, 26, 256 =ConvBlock=> P4: 26, 26, 512 =1*1conv2d=> P4: 26, 26, num_anchors * (5 + num_classes)
        out1 = self.yolo_head2(P4)
        # P5: 13, 13, 512 =ConvBlock=> P5: 13, 13, 1024 =1*1conv2d=> P5: 13, 13, num_anchors * (5 + num_classes)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
