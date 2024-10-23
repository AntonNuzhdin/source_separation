from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    SignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio
)
from torch import Tensor


class SISNRi:
    def __init__(self, *args, **kwargs) -> None:
        self.sisnr = ScaleInvariantSignalNoiseRatio()

    def __call__(self, predict: Tensor, target: Tensor, mixture: Tensor, **kwargs):
        metric = self.sisnr.to(predict.device)
        sisnr_pred = metric(predict, target).item()
        sisnr_mix = metric(mixture, target).item()
        sisnri = sisnr_pred - sisnr_mix
        return sisnri


class SISDR:
    def __init__(self, *args, **kwargs) -> None:
        self.sisdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, predict: Tensor, target: Tensor, **kwargs):
        metric = self.sisdr.to(predict.device)
        return metric(predict, target).item()


class PESQ:
    def __init__(self, fs: int, mode: str = 'wb', *args, **kwargs) -> None:
        self.pesq = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode)

    def __call__(self, predict: Tensor, target: Tensor, **kwargs):
        metric = self.pesq.to(predict.device)
        return metric(predict, target).item()


class SDRi:
    def __init__(self, *args, **kwargs) -> None:
        self.sdr = SignalDistortionRatio(*args, **kwargs)

    def __call__(self, predict: Tensor, target: Tensor, mixture: Tensor, **kwargs):
        metric = self.sdr.to(predict.device)
        sdr_pred = metric(predict, target).item()
        sdr_mix = metric(mixture, target).item()
        sdri = sdr_pred - sdr_mix
        return sdri


class STOI:
    def __init__(self, fs: int, extended: bool = False, *args, **kwargs) -> None:
        self.stoi = ShortTimeObjectiveIntelligibility(fs=fs, extended=extended, *args, **kwargs)

    def __call__(self, predict: Tensor, target: Tensor, **kwargs):
        metric = self.stoi.to(predict.device)
        return metric(predict, target).item()
