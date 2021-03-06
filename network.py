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
weights = torch.load('SweiNet_weights.pt')


def build_model(i):
    """Constructs SweiNet and loads a set of weights"""
    m = SweiNet(out_c=2, base_c=16, c_fact=(2, 4, 4))
    m.load_state_dict(weights[i])
    m.eval()
    return m


def get_model(use_ensemble):
    """Builds the ensemble or single model"""
    if use_ensemble:
        model = [build_model(i) for i in range(1, 31) if i != 18]
    else:
        model = [build_model(0)]
    return model


def run_model(input_, model, device='cpu'):
    """Run the model on preprocessed input
    Returns an array of estimated (m, sigma) values
    """
    displ = torch.from_numpy(input_['displ']).float().to(device)
    model = [m.to(device) for m in model]

    with torch.no_grad():
        z = [m(displ[:, None]) for m in model]
        z = torch.mean(torch.stack(z), dim=0)

    z[:, 0] = torch.exp(z[:, 0]) * input_['dxdt_factor']
    z[:, 1] = torch.exp(z[:, 1] / 2)
    z = z.cpu().numpy()
    return z


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
