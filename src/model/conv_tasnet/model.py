import torch
import torch.nn as nn

from src.model.conv_tasnet.utils import gLN
from src.model.conv_tasnet.separation import Separation
from src.model.lipreading.lipreading_model.main import get_visual_embeddings

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
        embed_dim
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
            n_sources=2,
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
        batch_size, channels, total_frames = input_tensor.size()
        remaining_frames = total_frames % self.enc_stride

        if remaining_frames > 0:
            input_tensor = F.pad(input_tensor, (0, self.enc_stride - remaining_frames))

        feature_frames = self.encoder(input_tensor)
        mask_output = self.separation(feature_frames, embed_video=embed_video) * feature_frames.unsqueeze(1)
        reshaped_masked_output = mask_output.view(batch_size * 2, self.encoder_dim, -1)

        decoded_output = self.decoder(reshaped_masked_output).view(batch_size, 2, input_tensor.size(2))

        if remaining_frames > 0:
            decoded_output = decoded_output[..., :-(self.enc_stride - remaining_frames)]

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
        use_visual=False,
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
            embed_dim
        )
        self.use_visual = use_visual

    def forward(self, mix_audio, s1_mouth, s2_mouth, **batch):
        if self.use_visual:
            s1_visual_embeddings = get_visual_embeddings(batch['s1_mouth'])
            s2_visual_embeddings = get_visual_embeddings(batch['s2_mouth'])
            embed_video = torch.stack((s1_visual_embeddings, s2_visual_embeddings))
        else:
            embed_video = None

        mix_audio = mix_audio.unsqueeze(1)
        separated = self.conv_tasnet(mix_audio, embed_video=embed_video)
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
