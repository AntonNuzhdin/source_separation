from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio
)
from src.metrics.base_metric import BaseMetric
import torch
from torch import Tensor
from src.metrics.utils import compute_metric


class SISNRi(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisnr = ScaleInvariantSignalNoiseRatio()

    def __call__(self, s1_audio, s2_audio, mix_audio, speaker_1,
                 speaker_2, **kwargs):
        metric = self.sisnr.to(s1_audio.device)
        return compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio)


class SISDRi(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, s1_audio, s2_audio, mix_audio, speaker_1,
                 speaker_2, **kwargs):
        metric = self.sisdr.to(s1_audio.device)
        return compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio)


class PESQ(BaseMetric):
    def __init__(self, fs, mode: str = 'wb', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode)

    def __call__(self, s1_audio, s2_audio, mix_audio, speaker_1,
                 speaker_2, **kwargs):
        metric = self.pesq.to(s1_audio.device)
        return compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio)


class SDRi(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        name = kwargs.pop('name', None)
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio(*args, **kwargs)

    def __call__(self, s1_audio, s2_audio, mix_audio, speaker_1,
                 speaker_2, **kwargs):
        metric = self.sdr.to(s1_audio.device)
        return compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio)


class STOI(BaseMetric):
    def __init__(self, fs, extended: bool = False, *args, **kwargs) -> None:
        name = kwargs.pop('name', None)
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=fs, extended=extended, *args, **kwargs)

    def __call__(self, s1_audio, s2_audio, mix_audio, speaker_1,
                 speaker_2, **kwargs):
        metric = self.stoi.to(s1_audio.device)
        return compute_metric(metric, s1_audio, s2_audio, speaker_1, speaker_2, mix_audio)
