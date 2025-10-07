import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# =========================
# Frozen BatchNorm Layer
# =========================
class FrozenBN(nn.Module):
    """
    Frozen BatchNorm layer to stabilize fine-tuning.
    The running statistics and affine parameters are fixed.
    """
    def __init__(self, num_channels: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.params_set = False

    def set_params(self, scale, bias, running_mean, running_var):
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.params_set = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.params_set, "Call set_params(...) before forward()"
        return torch.batch_norm(
            x, self.scale, self.bias, self.running_mean, self.running_var,
            training=False, momentum=self.momentum, eps=self.eps, cudnn_enabled=torch.backends.cudnn.enabled
        )

    def __repr__(self):
        return f"FrozenBN({self.num_channels})"


def freeze_bn(module: nn.Module):
    """
    Recursively replace all BatchNorm3d layers in a model with FrozenBN layers.
    """
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, nn.BatchNorm3d):
            frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
            frozen_bn.set_params(
                target_attr.weight.data, target_attr.bias.data,
                target_attr.running_mean, target_attr.running_var
            )
            setattr(module, attr_str, frozen_bn)
    for _, child in module.named_children():
        freeze_bn(child)


# =========================
# Bottleneck Block
# =========================
class Bottleneck(nn.Module):
    """
    3D ResNet bottleneck block with optional temporal convolution and stride.
    """
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int, downsample: Optional[nn.Module],
                 temp_conv: int, temp_stride: int):
        super().__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1),
            stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3),
            stride=(1, stride, stride), padding=(0, 1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# =========================
# I3Res50forStrokeOutcome Model
# =========================
class I3Res50forStrokeOutcome(nn.Module):
    """
    3D-ResNet50 backbone adapted for stroke outcome prediction.
    Uses a truncated version of I3D with dropout regularization.
    """
    def __init__(self, input_cha: int = 1, block: nn.Module = Bottleneck,
                 layers: List[int] = [3, 4, 6], num_classes: int = 1):
        super().__init__()
        self.inplanes = 64

        self.conv1_ = nn.Conv3d(
            input_cha, 64, kernel_size=(5, 7, 7),
            stride=(2, 2, 2), padding=(2, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1_ = nn.Linear(512 * round(block.expansion / 2), num_classes)

        self.drop = nn.Dropout(0.5)
        self.drop3D = nn.Dropout3d(0.5)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion, kernel_size=1,
                    stride=(temp_stride[0], stride, stride), bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0])]
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i]))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1_(x)))
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.drop3D(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.drop(x)
        x = x.flatten(1)
        return self.fc1_(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_single(x)


# =========================
# Model initialization helper
# =========================
def i3_res50forStrokeOutcome(input_cha: int = 1, num_classes: int = 1,
                             pretrained_path: Optional[str] = None) -> I3Res50forStrokeOutcome:
    """
    Initializes the I3Res50 model. Optionally loads pretrained weights.

    Args:
        input_cha: Number of input channels (e.g., 1 for single MRI sequence)
        num_classes: Output classes (1 for binary classification)
        pretrained_path: Optional path to pretrained 3D ResNet weights
    """
    net = I3Res50forStrokeOutcome(input_cha=input_cha, num_classes=num_classes)

    if pretrained_path and os.path.exists(pretrained_path):
        try:
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")
            model_dict = net.state_dict()
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(overlap_dict)
            net.load_state_dict(model_dict)
            print(f"[INFO] Loaded pretrained weights from: {pretrained_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load pretrained weights: {e}")

    return net
