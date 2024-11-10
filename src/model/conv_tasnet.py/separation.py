import torch
import torch.nn as nn
from src.model.conv_tasnet.utils import stack_layers, choice_activation


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
        msk_activate: str,
    ):
        super().__init__()

        self.n_sources = n_sources
        self.in_dim = in_dim

        self.input_norm = nn.GroupNorm(1, in_dim, eps=1e-7)
        self.input_conv = nn.Conv1d(in_dim, n_feats, kernel_size=1)

        self.conv_layers = nn.Sequential(
            *self._stack_layers(n_feats, n_hidden, kernel_size, n_layers, n_stacks)
        )

        self.output_prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(n_feats, in_dim * n_sources, kernel_size=1)
        self.mask_activate = self.choice_activation(msk_activate)

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
