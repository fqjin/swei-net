"""
Copyright 2022 Felix Q. Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Pre-activation Residual Block"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, in_c, 3, bias=False,
                               padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.conv2 = nn.Conv2d(in_c, out_c, 3, bias=False,
                               padding=1, stride=stride)
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(stride))
        if in_c != out_c:
            downsample.append(nn.Conv2d(in_c, out_c, 1, bias=False))
        self.downsample = nn.Sequential(*downsample)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += identity
        return x


class SweiNet(nn.Module):
    """A pre-activation ResNet for processing SWEI data
    Input shape: (16, 100)
    Output shape: (out_c, 1)
    """
    def __init__(
        self,
        out_c=2,
        base_c=16,
        c_fact=(2, 4, 4),
    ):
        super().__init__()
        self.out_c = out_c
        self.base_c = base_c
        self.c_fact = c_fact

        self.conv1 = nn.Conv2d(1, base_c, 5, bias=False, padding=0, stride=1)
        self.block1 = ResBlock(base_c, c_fact[0] * base_c, stride=2)
        self.block2 = ResBlock(c_fact[0] * base_c, c_fact[1] * base_c, stride=2)
        self.block3 = ResBlock(c_fact[1] * base_c, c_fact[2] * base_c, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        self.pooled_c = 4 * c_fact[2] * base_c
        self.fc = nn.Linear(self.pooled_c, out_c, bias=True)

        # Zero-initialize the last BN in each residual branch,
        #   according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):  # 1 x 16 x 100
        x = self.conv1(x)  # c x 12 x 96
        x = self.block1(x)  # c x 6 x 48
        x = self.block2(x)  # c x 3 x 24
        x = self.block3(x)  # c x 3 x 24
        x = self.pool(x).view(-1, self.pooled_c)  # c * 1 * 4
        x = self.fc(x)  # out_c
        return x


if __name__ == '__main__':
    for m in [
        SweiNet(out_c=2, base_c=16, c_fact=(2, 4, 4)),  # 112210
    ]:
        params = sum(p.numel() for p in m.parameters())
        print(params)
