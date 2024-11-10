import torch
import torch.nn as nn

from src.model.conv_tasnet.utils import choice_activation, gLN, Conv1DBlock


class Separation(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_sources: int,
        kernel_size: int,
        n_feats: int,
        n_hidden: int,
        n_layers: int,
        n_stacks: int,
        # msk_activate: str,
    ):
        super().__init__()

        self.n_feats = n_feats
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_stacks = n_stacks

        self.n_sources = n_sources
        self.in_dim = in_dim

        self.input_norm = gLN(dimension=in_dim)
        self.input_conv = nn.Conv1d(in_dim, n_feats, kernel_size=1)

        self.conv_layers = nn.Sequential(
            *self.stack_layers()
        )

        self.output_prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(n_feats, in_dim * n_sources, kernel_size=1)
        self.mask_activate = nn.ReLU()

    def stack_layers(self):
        layers = []
        for stack in range(self.n_stacks):
            for layer in range(self.n_layers):
                dilation = 2 ** layer
                layers.append(
                    Conv1DBlock(
                        input_channels=self.n_feats,
                        hidden_channels=self.n_hidden,
                        kernel_size=self.kernel_size,
                        dilation=dilation,
                        padding=dilation,
                        no_residual=(stack == self.n_stacks - 1 and layer == self.n_layers - 1),
                    )
                )
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        feats = self.input_norm(x)
        feats = self.input_conv(feats)

        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)

        return output.view(batch_size, self.n_sources, self.in_dim, -1)
