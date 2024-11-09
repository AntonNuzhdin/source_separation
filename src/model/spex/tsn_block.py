import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        in_channels_emb, 
        out_channels
    ):
        super().__init__()
        
        self.tcn_seq = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels + in_channels_emb, out_channels=out_channels, kernel_size=1
            ), 
            nn.PReLU(),
            nn.
        )
        self.conv1_1 = 



    def forward(self, x, speaker_embedding=None):
        if speaker_embedding:
            input = torch.cat([x, speaker_embedding], dim=1)

