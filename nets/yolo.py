from collections import OrderedDict

import torch
import torch.nn as nn

from nets.efficientnet import EfficientNet as EffNet


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
        out_feats = [feature_maps[2],feature_maps[3],feature_maps[4]]
        return out_feats


def cbl(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def spp(spp_in):
    maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # padding = (k - 1) // 2
    maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
    maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
    return torch.cat((maxpool1(spp_in), maxpool2(spp_in), maxpool3(spp_in), spp_in), 1)

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        cbl(in_filters, filters_list[0], 1),
        cbl(filters_list[0], filters_list[1], 3),
        cbl(filters_list[1], filters_list[0], 1),
        cbl(filters_list[0], filters_list[1], 3),
        cbl(filters_list[1], filters_list[0], 1),
        cbl(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
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
        filter_list = [out_filters[-1], int(out_filters[-1] / 2), int(out_filters[-1] / 4), int(out_filters[-1] / 8)]

        # Neck layers
        self.spp_in = nn.Sequential(
            cbl(filter_list[0], filter_list[1], 1),
            cbl(filter_list[1], filter_list[0], 1),
            cbl(filter_list[0], filter_list[1], 1)
        )

        self.spp_out = nn.Sequential(
            cbl(int(out_filters[-1] * 2), filter_list[1], 1),
            cbl(filter_list[1], filter_list[0], 1),
            cbl(filter_list[0], filter_list[1], 1)
        )

        self.cbl2 = cbl(filter_list[1], filter_list[2], 1)
        self.cbl3 = cbl(filter_list[2], filter_list[3], 1)
        self.upsample = nn.Upsample(scale_factor=2)

        # Last layers
        self.last_layer0_conv = cbl(filter_list[2], filter_list[3], 1)
        self.last_layers0     = make_last_layers([filter_list[3], filter_list[2]], filter_list[2],
                                                 len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv = cbl(filter_list[3], filter_list[2], 3, stride=2)
        self.last_layers1     = make_last_layers([filter_list[2], filter_list[1]], filter_list[1],
                                                 len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv = cbl(filter_list[2], filter_list[1], 3, stride=2)
        self.last_layers2     = make_last_layers([filter_list[1], filter_list[0]], filter_list[0],
                                                 len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52, 52, 40
        #   26, 26, 112
        #   13, 13, 320
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        spp_in = self.spp_in(x0)
        spp_out = spp(spp_in)
        panet_in = self.spp_out(spp_out)

        panet_branch0 = self.cbl2(panet_in)
        panet_branch0 = self.upsample(panet_branch0)
        panet_branch1 = self.cbl2(x1)
        panet_branch01 = torch.cat((panet_branch0, panet_branch1), 1)
        panet_branch01 = self.last_layers1[:5](panet_branch01)

        in_small = self.cbl3(panet_branch01)
        in_small = self.upsample(in_small)
        panet_branch2 = self.cbl3(x2)
        in_small = torch.cat((in_small, panet_branch2), 1)
        in_small = self.last_layers0[:5](in_small)
        out_small = self.last_layers0[5:](in_small)

        in_medium = self.last_layer1_conv(in_small)
        in_medium = torch.cat((in_medium, panet_branch01), 1)
        in_medium = self.last_layers1[:5](in_medium)
        out_medium = self.last_layers1[5:](in_medium)

        in_large = self.last_layer2_conv(in_medium)
        in_large = torch.cat((in_large, panet_in), 1)
        in_large = self.last_layers2[:5](in_large)
        out_large = self.last_layers2[5:](in_large)

        return out_small, out_medium, out_large
