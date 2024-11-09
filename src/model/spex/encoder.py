import torch
import torch.nn as nn


class TwinSpeechEncoder(nn.Module):
    def __init__(
        self,
        short_filter_len,
        middle_filter_len,
        long_filter_len,
        channels_out,
    ):
        super().__init__()

        self.stride = short_filter_len // 2
        self.short_1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels_out, kernel_size=short_filter_len, stride=self.stride),
            nn.ReLU()
        )
        self.middle_1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels_out, kernel_size=middle_filter_len, stride=self.stride),
            nn.ReLU()
        )
        self.long_1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels_out, kernel_size=long_filter_len, stride=self.stride),
            nn.ReLU()
        )

        self.padding_middle = nn.ConstantPad1d(padding=(0, middle_filter_len - short_filter_len), value=0)
        self.padding_long = nn.ConstantPad1d(padding=(0, middle_filter_len - long_filter_len), value=0)

    def forward(self, wav):
        # wav - [B, L]
        wav_short = self.short_1d(wav)
        wav_short = self.padding_middle(wav_short)
        wav_middle = self.middle_1d(wav_short)
        wav_middle = self.padding_long(wav_middle)
        wav_long = self.long_1d(wav_middle)
        out = torch.cat([wav_short, wav_middle, wav_long], dim=1)
        return out
