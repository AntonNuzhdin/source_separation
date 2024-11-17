import torch
import torch.nn as nn

from src.model.conv_tasnet.utils import gLN, Conv1DBlock
from src.model.conv_tasnet.add_vision_embeds import MultimodaFusion
import torch.nn.functional as F


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
        embed_dim: int,
        fused_dim: int = 128
    ):
        super().__init__()

        self.n_feats = n_feats
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_stacks = n_stacks

        self.n_sources = n_sources
        self.in_dim = in_dim
        self.embed_dim = embed_dim


        self.input_proc = nn.Sequential(
            gLN(dimension=in_dim),
            nn.Conv1d(in_dim, n_feats, kernel_size=1)
        )

        self.video_proc = nn.LSTM(
            input_size=embed_dim,
            hidden_size=n_feats,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        self.multimoda_fusion = MultimodaFusion(audio_dim=n_feats, visual_dim=n_feats * 2, fused_dim=fused_dim)


        self.conv_layers = nn.Sequential(
            *self.stack_layers()
        )

        self.out = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(n_feats, in_dim * n_sources, kernel_size=1),
            nn.Sigmoid()
        )

    def stack_layers(self):
        layers = []
        for stack in range(self.n_stacks):
            for layer in range(self.n_layers):
                layers.append(
                    Conv1DBlock(
                        input_channels=self.n_feats,
                        hidden_channels=self.n_hidden,
                        kernel_size=self.kernel_size,
                        dilation=2 ** layer,
                        padding=2 ** layer,
                        no_residual=False,
                    )
                )
        layers.append(
            Conv1DBlock(
                input_channels=self.n_feats,
                hidden_channels=self.n_hidden,
                kernel_size=self.kernel_size,
                dilation=2 ** (self.n_layers - 1),
                padding=2 ** (self.n_layers - 1),
                no_residual=True,
            )
        )
        return layers

    def forward(self, inputs, embed_video=None):
        batch_size = inputs.size(0)
        audio_inputs = self.input_proc(inputs)

        visual_features = None
        if embed_video is not None:
            visual_features, _ = self.video_proc(embed_video)
            if visual_features.size(-1) != audio_inputs.size(-1):
                visual_features = F.interpolate(
                    visual_features.permute(0, 2, 1), size=audio_inputs.size(-1), mode='nearest'
                ).permute(0, 2, 1)

        accumulated_output = 0.0
        for i, conv_block in enumerate(self.conv_layers):
            if i == 8 and embed_video is not None:
                residual_output, skip_connection = conv_block(audio_inputs)
                residual_output = self.multimoda_fusion(residual_output, visual_features)
                audio_inputs = audio_inputs + residual_output
                accumulated_output = accumulated_output + skip_connection

            else:
                residual_output, skip_connection = conv_block(audio_inputs)

                if residual_output is not None:
                    audio_inputs = audio_inputs + residual_output

                accumulated_output = accumulated_output + skip_connection

        final_output = self.out(accumulated_output)
        return final_output.view(batch_size, self.n_sources, self.in_dim, -1)