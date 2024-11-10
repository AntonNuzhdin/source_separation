import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.audio import (
    ScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalDistortionRatio
)


class SISNR_LOSS(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisnr = ScaleInvariantSignalNoiseRatio()

    def forward(self, s1_audio: Tensor, s2_audio: Tensor, output_spk1: Tensor,
                output_spk2: Tensor, **kwargs):
        metric = self.sisnr.to(s1_audio.device)
        sisnr_s1_s1 = metric(s1_audio, output_spk1).item()
        sisnr_s1_s2 = metric(s1_audio, output_spk2).item()
        sisnr_s2_s1 = metric(s2_audio, output_spk1).item()
        sisnr_s2_s2 = metric(s2_audio, output_spk2).item()

        loss = torch.maximum(torch.tensor((sisnr_s1_s1 + sisnr_s2_s2) / 2),
                             torch.tensor((sisnr_s1_s2 + sisnr_s2_s1) / 2))
        return loss.mean()


class SISDR_LOSS(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, s1_audio: Tensor, s2_audio: Tensor, output_spk1: Tensor,
                output_spk2: Tensor, **kwargs):
        metric = self.sisdr.to(s1_audio.device)
        sisnr_s1_s1 = metric(s1_audio, output_spk1).item()
        sisnr_s1_s2 = metric(s1_audio, output_spk2).item()
        sisnr_s2_s1 = metric(s2_audio, output_spk1).item()
        sisnr_s2_s2 = metric(s2_audio, output_spk2).item()

        loss = torch.maximum(torch.tensor((sisnr_s1_s1 + sisnr_s2_s2) / 2),
                             torch.tensor((sisnr_s1_s2 + sisnr_s2_s1) / 2))
        return loss.mean()
