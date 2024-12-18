import torch.nn as nn

from models.gmflow_model.trident_conv import MultiScaleTridentConv


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            stride=stride,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        norm_layer=nn.InstanceNorm2d,
        num_output_scales=1, # 训练中设置为2
        **kwargs,
    ):
        super(CNNEncoder, self).__init__()
        self.num_branch = num_output_scales  # 训练中设置为2

        feature_dims = [64, 96, 128]

        strides = [2, 1, 2]
        # strides = [1, 1, 1]

        self.conv1 = nn.Conv2d(
            3, feature_dims[0], kernel_size=7, stride=strides[0], padding=3, bias=False
        )  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(
            feature_dims[0], stride=strides[1], norm_layer=norm_layer
        )  # 1/2
        self.layer2 = self._make_layer(
            feature_dims[1], stride=strides[2], norm_layer=norm_layer
        )  # 1/4

        # highest resolution 1/4 or 1/8
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(
            feature_dims[2],
            stride=stride,
            norm_layer=norm_layer,
        )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError

            self.trident_conv = MultiScaleTridentConv(
                output_dim,
                output_dim,
                kernel_size=3,
                strides=strides,
                paddings=1,
                num_branch=self.num_branch,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(
            self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation # 第一层残差块的作用是调整输入特征的通道数和空间尺寸，使其匹配新的输出特征维度
        )
        layer2 = ResidualBlock(
            dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation # 第二层残差块的作用是进一步处理特征，同时保持通道数和空间尺寸不变。
        )

        layers = (layer1, layer2)

        self.in_planes = dim 
        # 将当前层的输出通道数 dim 保存到 self.in_planes 中，作为后续层的输入通道数。
        # 这是构建层级式网络（例如 ResNet）的常见做法，因为每一层的输出需要成为下一层的输入。
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x) # [2*B*(2T-2), C, H, W] -->[2*B*(2T-2), 64, H/2, W/2]
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2 [2*B*(2T-2), 64, H/2, W/2] --> [2*B*(2T-2), 64, H/2, W/2]
        x = self.layer2(x)  # 1/4 [2*B*(2T-2), 64, H/2, W/2] --> [2*B*(2T-2), 96, H/4, W/4]
        x = self.layer3(x)  # 1/8 or 1/4  [2*B*(2T-2), 96, H/4, W/4] --> [2*B*(2T-2), 128, H/4, W/4]

        x = self.conv2(x) # [2*B*(2T-2), 128, H/4, W/4] --> [2*B*(2T-2), 128, H/4, W/4]

        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)  # high to low res [2*B*(2T-2), 128, H/4, W/4]；[2*B*(2T-2), 128, H/8, W/8]
        else:
            out = [x]

        return out
