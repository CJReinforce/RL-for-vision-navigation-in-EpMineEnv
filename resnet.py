import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:1'


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3,
                 stride=stride, groups=groups),
            Conv(mid_channels, out_channels,
                 kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Img_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages = nn.Sequential(*[
            self._make_stage(64, 256, down_sample=False, num_blocks=3),
            self._make_stage(256, 512, down_sample=True, num_blocks=4),
            self._make_stage(512, 1024, down_sample=True, num_blocks=6),
            self._make_stage(1024, 2048, down_sample=True, num_blocks=3),
        ])
        self.head = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2048, num_classes)
        ])

        self.pose_mlp = nn.ModuleList(
            [MLP(num_classes, 1, num_classes * 4) for _ in range(4)]
        )

        self.position_mlp = nn.ModuleList(
            [MLP(num_classes, 1, num_classes * 4) for _ in range(3)]
        )

    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels,
                             down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(
                out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        shape = x.shape
        if shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        assert x.shape[-3:] == (3, 224, 224)
        # Resnet 50
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        # Pose and position caluculation using MLP
        for i in range(4):
            pose_outputs = self.pose_mlp[i](x) if i == 0 else torch.cat(
                (pose_outputs, self.pose_mlp[i](x)), 1)
        for i in range(3):
            position_outputs = self.position_mlp[i](x) if i == 0 else torch.cat(
                (position_outputs, self.position_mlp[i](x)), 1)

        outputs = torch.cat((pose_outputs, position_outputs), 1)
        return outputs
