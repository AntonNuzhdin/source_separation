import torch
import torch.nn as nn

from src.model.conv_tasnet.utils import gLN
from src.model.conv_tasnet.separation import Separation


class Model(nn.Module):
    def __init__(
        self,
        encoder_kernel_size,
        encoder_dim,
        separation_kernel_size,
        separation_n_feats,
        separation_n_hidden,
        separation_n_layers,
        separation_n_stacks,
    ):
        super().__init__()
        self.encoder_kernel_size = encoder_kernel_size
        self.enc_stride = encoder_kernel_size // 2
        self.encoder_dim = encoder_dim
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_dim,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            padding=encoder_kernel_size // 2,
        )

        self.separation = Separation(
            in_dim=encoder_dim,
            n_sources=2,
            kernel_size=separation_kernel_size,
            n_feats=separation_n_feats,
            n_hidden=separation_n_hidden,
            n_layers=separation_n_layers,
            n_stacks=separation_n_stacks
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_dim,
            out_channels=1,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            padding=encoder_kernel_size // 2,
        )

    def forward(self, x):
        batch_size, num_channels, num_frames = x.shape
        num_remainings = num_frames - (num_frames // self.enc_stride * self.enc_stride)

        if num_remainings > 0:
            num_paddings = self.enc_stride - num_remainings
            pad = torch.zeros(
                batch_size,
                num_channels,
                num_paddings,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], 2)

        num_padded_frames = x.shape[2]
        feats = self.encoder(x)
        masked = self.separation(feats) * feats.unsqueeze(1)
        masked = masked.view(batch_size * 2, self.encoder_dim, -1)
        decoded = self.decoder(masked)
        output = decoded.view(batch_size, 2, num_padded_frames)

        if num_remainings > 0:
            output = output[..., :-num_paddings]

        return output


class ConvTasNet(nn.Module):
    def __init__(
        self,
        encoder_kernel_size=16,
        encoder_dim=512,
        separation_kernel_size=3,
        separation_n_feats=128,
        separation_n_hidden=512,
        separation_n_layers=8,
        separation_n_stacks=3,
    ):
        super().__init__()
        self.conv_tasnet = Model(
            encoder_kernel_size,
            encoder_dim,
            separation_kernel_size,
            separation_n_feats,
            separation_n_hidden,
            separation_n_layers,
            separation_n_stacks
        )

    def forward(self, mix_audio, **batch):
        mix_audio = mix_audio.unsqueeze(1)
        separated = self.conv_tasnet(mix_audio)
        return {
            'speaker_1': separated[:, 0, :].squeeze(1),
            'speaker_2': separated[:, 1, :].squeeze(1)
        }
