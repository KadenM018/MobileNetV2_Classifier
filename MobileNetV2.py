import torch.nn as nn
import torch


class InvertedResidualBottleneck(nn.Module):
    def __init__(self, expand_factor, in_channels, out_channels, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)

        expanded_depth = in_channels * expand_factor

        # bool to apply a residual connection when the stride equals 1
        self.residual_conn = False
        if stride == 1 and in_channels == out_channels:
            self.residual_conn = True

        self.irb_model = nn.Sequential(
            # 1x1 convolution
            nn.Conv2d(in_channels, expanded_depth, (1, 1), 1),
            nn.BatchNorm2d(expanded_depth),
            nn.ReLU6(),

            # 3x3 depthwise convolution
            nn.Conv2d(expanded_depth, expanded_depth, (3, 3), stride, 1, groups=expanded_depth),
            nn.BatchNorm2d(expanded_depth),
            nn.ReLU6(),

            # 1x1 convolution
            nn.Conv2d(expanded_depth, out_channels, (1, 1), 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual_conn:
            return x + self.irb_model(x)
        else:
            return self.irb_model(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential()

        bottleneck_arch = [
            # expand factor, output channels, repetitions, stride of first conv
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.model.add_module('conv2d_0', nn.Conv2d(in_channels, 32, (3, 3), 2))

        i_channels = 32
        for i in range(0, len(bottleneck_arch)):
            e_factor = bottleneck_arch[i][0]
            o_channels = bottleneck_arch[i][1]
            repetitions = bottleneck_arch[i][2]
            stride = bottleneck_arch[i][3]

            # first layer uses the stride variable for stride
            self.model.add_module(f'bottleneck_first_{i}',
                                  InvertedResidualBottleneck(e_factor, i_channels, o_channels, stride))

            # Every layer after first uses stride of 1
            for j in range(1, repetitions):
                self.model.add_module(f'bottleneck_{i}_{j}',
                                      InvertedResidualBottleneck(e_factor, o_channels, o_channels, 1))

            i_channels = o_channels

        self.model.add_module('conv2d_1', nn.Conv2d(i_channels, 1280, (1, 1), 1))
        self.model.add_module('avgpooling', nn.AvgPool2d((7, 7), 1))
        # self.model.add_module('conv2d_2', nn.Conv2d(1280, in_channels, (1, 1), 1))

        self.classifier = nn.Sequential(nn.Linear(1280, num_classes))

    def forward(self, x):
        features = self.model(x)
        return self.classifier(torch.flatten(features, start_dim=1, end_dim=-1))
