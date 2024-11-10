from src.model.conv_tasnet.model import Conv1DBlock
import torch.nn as nn


def stack_layers(self, n_feats, n_hidden, kernel_size, n_layers, n_stacks):
    layers = []
    for stack in range(n_stacks):
        for layer in range(n_layers):
            dilation = 2 ** layer
            layers.append(
                Conv1DBlock(
                    io_channels=n_feats,
                    hidden_channels=n_hidden,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation,
                    no_residual=(stack == n_stacks - 1 and layer == n_layers - 1),
                )
            )
    return layers


def choice_activation(self, msk_activate: str):
    if msk_activate == "sigmoid":
        return nn.Sigmoid()
    elif msk_activate == "relu":
        return nn.ReLU()
