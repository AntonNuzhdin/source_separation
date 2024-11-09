from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    SignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio
)
from torch import Tensor
import torch
from src.metrics.base_metric import BaseMetric


class SISNRi(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisnr = ScaleInvariantSignalNoiseRatio()

    def __call__(self, s1_audio: Tensor, s2_audio: Tensor, mix_audio: Tensor, output_spk1: Tensor,
                 output_spk2: Tensor, **kwargs):
        metric = self.sisnr.to(s1_audio.device)
        sisnr_s1_s1 = metric(s1_audio, output_spk1).item()
        sisnr_s1_s2 = metric(s1_audio, output_spk2).item()
        sisnr_s2_s1 = metric(s2_audio, output_spk1).item()
        sisnr_s2_s2 = metric(s2_audio, output_spk2).item()

        sisnr_mix_s1 = metric(s1_audio, mix_audio).item()
        sisnr_mix_s2 = metric(s2_audio, mix_audio).item()

        sisnri_s1 = sisnr_s1_s1 - sisnr_mix_s1
        sisnri_s2 = sisnr_s2_s2 - sisnr_mix_s2

        loss = torch.maximum(torch.tensor((sisnri_s1 + sisnri_s2) / 2), torch.tensor((sisnr_s1_s2 + sisnr_s2_s1) / 2))
        return loss.mean()


class SISDR(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, s1_audio: Tensor, s2_audio: Tensor, mix_audio: Tensor, output_spk1: Tensor,
                 output_spk2: Tensor, **kwargs):
        metric = self.sisdr.to(s1_audio.device)
        sisdr_s1_s1 = metric(s1_audio, output_spk1).item()
        sisdr_s1_s2 = metric(s1_audio, output_spk2).item()
        sisdr_s2_s1 = metric(s2_audio, output_spk1).item()
        sisdr_s2_s2 = metric(s2_audio, output_spk2).item()

        sisdr_mix_s1 = metric(s1_audio, mix_audio).item()
        sisdr_mix_s2 = metric(s2_audio, mix_audio).item()

        sisdri_s1 = sisdr_s1_s1 - sisdr_mix_s1
        sisdri_s2 = sisdr_s2_s2 - sisdr_mix_s2

        loss = torch.maximum(torch.tensor((sisdri_s1 + sisdri_s2) / 2), torch.tensor((sisdr_s1_s2 + sisdr_s2_s1) / 2))
        return loss.mean()


class PESQ(BaseMetric):
    def __init__(self, fs: int, mode: str = 'wb', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode)

    def __call__(self, s1_audio: Tensor, s2_audio: Tensor, mix_audio: Tensor, output_spk1: Tensor,
                 output_spk2: Tensor, **kwargs):
        metric = self.pesq.to(s1_audio.device)
        pesq_s1_s1 = metric(s1_audio, output_spk1).item()
        pesq_s1_s2 = metric(s1_audio, output_spk2).item()
        pesq_s2_s1 = metric(s2_audio, output_spk1).item()
        pesq_s2_s2 = metric(s2_audio, output_spk2).item()

        pesq_mix_s1 = metric(s1_audio, mix_audio).item()
        pesq_mix_s2 = metric(s2_audio, mix_audio).item()

        pesqi_s1 = pesq_s1_s1 - pesq_mix_s1
        pesqi_s2 = pesq_s2_s2 - pesq_mix_s2

        loss = torch.maximum(torch.tensor((pesqi_s1 + pesqi_s2) / 2), torch.tensor((pesq_s1_s2 + pesq_s2_s1) / 2))
        return loss.mean()


class SDRi(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio(*args, **kwargs)

    def __call__(self, s1_audio: Tensor, s2_audio: Tensor, mix_audio: Tensor, output_spk1: Tensor,
                 output_spk2: Tensor, **kwargs):
        metric = self.sdr.to(s1_audio.device)
        sdr_s1_s1 = metric(s1_audio, output_spk1).item()
        sdr_s1_s2 = metric(s1_audio, output_spk2).item()
        sdr_s2_s1 = metric(s2_audio, output_spk1).item()
        sdr_s2_s2 = metric(s2_audio, output_spk2).item()

        sdr_mix_s1 = metric(s1_audio, mix_audio).item()
        sdr_mix_s2 = metric(s2_audio, mix_audio).item()

        sdri_s1 = sdr_s1_s1 - sdr_mix_s1
        sdri_s2 = sdr_s2_s2 - sdr_mix_s2

        loss = torch.maximum(torch.tensor((sdri_s1 + sdri_s2) / 2), torch.tensor((sdr_s1_s2 + sdr_s2_s1) / 2))
        return loss.mean()


class STOI(BaseMetric):
    def __init__(self, fs: int, extended: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=fs, extended=extended, *args, **kwargs)

    def __call__(self, s1_audio: Tensor, s2_audio: Tensor, mix_audio: Tensor, output_spk1: Tensor,
                 output_spk2: Tensor, **kwargs):
        metric = self.stoi.to(s1_audio.device)
        stoi_s1_s1 = metric(s1_audio, output_spk1).item()
        stoi_s1_s2 = metric(s1_audio, output_spk2).item()
        stoi_s2_s1 = metric(s2_audio, output_spk1).item()
        stoi_s2_s2 = metric(s2_audio, output_spk2).item()

        stoi_mix_s1 = metric(s1_audio, mix_audio).item()
        stoi_mix_s2 = metric(s2_audio, mix_audio).item()

        stoii_s1 = stoi_s1_s1 - stoi_mix_s1
        stoii_s2 = stoi_s2_s2 - stoi_mix_s2

        loss = torch.maximum(torch.tensor((stoii_s1 + stoii_s2) / 2), torch.tensor((stoi_s1_s2 + stoi_s2_s1) / 2))
        return loss.mean()
