import torch
import torch.nn as nn


class feature_module(nn.Module):
    def __init__(self, in_dim):
        super(feature_module, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_dim, 256, kernel_size=3, padding=1, groups=4, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(inplace=True),
        )

        self.mem_f = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)
        self.mem_b = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)

        self.layer2_f = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, groups=4, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ReLU(inplace=True),
        )

        self.layer2_b = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, groups=4, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ReLU(inplace=True),
        )

        self.layer2_d = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, padding=0, groups=4, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # flatten param for multi GPU training
        self.mem_f.flatten_parameters()
        self.mem_b.flatten_parameters()

        # layer 1
        x = self.layer1(x)  # [B,C,T]

        # memory
        x_f, _ = self.mem_f(x.permute(2, 0, 1))  # [T,B,C]
        x_b, _ = self.mem_b(x.permute(2, 0, 1).flip(dims=[0]))  # [T,B,C]

        # layer 2
        x_f = self.layer2_f(x_f.permute(1, 2, 0))
        x_b = self.layer2_b(x_b.flip(dims=[0]).permute(1, 2, 0))
        x = self.layer2_d(x)

        x = torch.cat((x_f, x, x_b), dim=1)

        # layer 3
        x = self.layer3(x)  # [B,C,T]
        return x
