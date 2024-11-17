import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        embed_dim,
        n_sources
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
            bias=False,
        )

        self.separation = Separation(
            in_dim=encoder_dim,
            n_sources=n_sources,
            kernel_size=separation_kernel_size,
            n_feats=separation_n_feats,
            n_hidden=separation_n_hidden,
            n_layers=separation_n_layers,
            n_stacks=separation_n_stacks,
            embed_dim=embed_dim
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_dim,
            out_channels=1,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            padding=encoder_kernel_size // 2,
            bias=False,
        )

    def forward(self, input_tensor, embed_video=None):
        if embed_video == None:
            batch_size, _, _ = input_tensor.size()

            feature_frames = self.encoder(input_tensor)
            mask_output = self.separation(feature_frames, embed_video=embed_video) * feature_frames.unsqueeze(1)
            reshaped_masked_output = mask_output.view(batch_size * 2, self.encoder_dim, -1)

            decoded_output = self.decoder(reshaped_masked_output).view(batch_size, 2, input_tensor.size(2))
            return decoded_output

        batch_size, _, _ = input_tensor.size()

        feature_frames = self.encoder(input_tensor)
        mask_output = self.separation(feature_frames, embed_video=embed_video) * feature_frames.unsqueeze(1)
        reshaped_masked_output = mask_output.view(batch_size, self.encoder_dim, -1)

        decoded_output = self.decoder(reshaped_masked_output).view(batch_size, 1, input_tensor.size(2))

        return decoded_output


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
        embed_dim=512,
        n_sources=1,
        use_visual=True,
    ):
        super().__init__()
        self.conv_tasnet = Model(
            encoder_kernel_size,
            encoder_dim,
            separation_kernel_size,
            separation_n_feats,
            separation_n_hidden,
            separation_n_layers,
            separation_n_stacks,
            embed_dim,
            n_sources,
        )
        self.use_visual = use_visual

    def forward(self, mix_audio, emb_s1, emb_s2, **batch):
        mix_audio = mix_audio.unsqueeze(1)
        if self.use_visual:
            separated_1 = self.conv_tasnet(mix_audio, embed_video=emb_s1)
            separated_2 = self.conv_tasnet(mix_audio, embed_video=emb_s2)
            return {
                'speaker_1': separated_1.squeeze(1),
                'speaker_2': separated_2.squeeze(1)
            }
        separated = self.conv_tasnet(mix_audio, embed_video=None)

        return {
            'speaker_1': separated[:, 0, :].squeeze(1),
            'speaker_2': separated[:, 1, :].squeeze(1)
        }

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
