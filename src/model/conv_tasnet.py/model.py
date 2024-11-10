import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        input_channles,
        hidden_channels,
        kernel_size,
    ):
        super().__init__()


class ConvTasNet(nn.Module):
    def __init__(
        self,
        N,
        L,
        B,
        H,
        Sc,
        P,
        X,
        R
    ):
        super().__init__()
        self.encoder = nn.Conv1d()
