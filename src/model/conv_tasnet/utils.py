import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        dilation,
        padding,
        no_residual,
    ):
        super().__init__()
        self.no_residual = no_residual
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=1),
            nn.PReLU(),
            gLN(dimension=hidden_channels),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels
            ),
            nn.PReLU(),
            gLN(dimension=hidden_channels)
        )
        if no_residual:
            self.conv_output = None
        else:
            self.conv_output = nn.Conv1d(in_channels=hidden_channels, out_channels=input_channels, kernel_size=1)
        self.skip_output = nn.Conv1d(in_channels=hidden_channels, out_channels=input_channels, kernel_size=1)

    def forward(self, x):
        seq = self.seq(x)
        if self.no_residual:
            conv_output = None
        else:
            conv_output = self.conv_output(seq)
        skip_output = self.skip_output(seq)
        return conv_output, skip_output


class gLN(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

        self.w = nn.Parameter(torch.ones(self.dimension, 1))
        self.b = nn.Parameter(torch.zeros(self.dimension, 1))

    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        sqrt_var = torch.sqrt(torch.mean((x - mean) ** 2, (1, 2), keepdim=True) + 1e-6)
        return self.w * (x - mean) / sqrt_var + self.b


def choice_activation(self, msk_activate: str):
    if msk_activate == "sigmoid":
        return nn.Sigmoid()
    elif msk_activate == "relu":
        return nn.ReLU()
