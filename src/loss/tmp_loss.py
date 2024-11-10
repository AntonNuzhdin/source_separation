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

    def forward(self, s1_audio: Tensor, s2_audio: Tensor, speaker_1: Tensor,
                speaker_2: Tensor, **kwargs):
        metric = self.sisnr.to(s1_audio.device)

        sisnr_s1_s1 = metric(s1_audio, speaker_1)
        sisnr_s1_s2 = metric(s1_audio, speaker_2)
        sisnr_s2_s1 = metric(s2_audio, speaker_1)
        sisnr_s2_s2 = metric(s2_audio, speaker_2)

        loss = torch.maximum((sisnr_s1_s1 + sisnr_s2_s2) / 2,
                             (sisnr_s1_s2 + sisnr_s2_s1) / 2)

        return {
            "loss": loss.mean()
        }


class SISDR_LOSS(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, s1_audio: Tensor, s2_audio: Tensor, speaker_1: Tensor,
                speaker_2: Tensor, **kwargs):
        metric = self.sisdr.to(s1_audio.device)

        sisdr_s1_s1 = metric(s1_audio, speaker_1)
        sisdr_s1_s2 = metric(s1_audio, speaker_2)
        sisdr_s2_s1 = metric(s2_audio, speaker_1)
        sisdr_s2_s2 = metric(s2_audio, speaker_2)

        loss = torch.maximum((sisdr_s1_s1 + sisdr_s2_s2) / 2,
                             (sisdr_s1_s2 + sisdr_s2_s1) / 2)

        return {
            "loss": loss.mean()
        }
