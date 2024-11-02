import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SISDRLoss(nn.Module):
    def __init__(self):
        super(SISDRLoss, self).__init__()

    def forward(self, est_source: Tensor, target_source: Tensor) -> Tensor:
        eps = 1e-8
        est_source = est_source - torch.mean(est_source, dim=-1, keepdim=True)
        target_source = target_source - torch.mean(target_source, dim=-1, keepdim=True)

        dot_product = torch.sum(est_source * target_source, dim=-1, keepdim=True)
        target_energy = torch.sum(target_source ** 2, dim=-1, keepdim=True)

        projection = dot_product / (target_energy + eps) * target_source
        noise = est_source - projection

        sdr = 10 * torch.log10((torch.sum(projection ** 2, dim=-1) + eps)
                               / (torch.sum(noise ** 2, dim=-1) + eps))

        return -torch.mean(sdr)


class CrossEntropyLossWrapper(nn.CrossEntropyLoss):
    def forward(self, inputs: Tensor, targets: Tensor, **batch) -> Tensor:
        loss = super().forward(inputs, targets)
        return {"loss_ce": loss}


class CombinedLoss(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super(CombinedLoss, self).__init__()
        self.sisdr_loss = SISDRLoss()
        self.cross_entropy_loss = CrossEntropyLossWrapper()

        self.gamma = gamma

    def forward(self, est_source: Tensor, target_source: Tensor, logits: Tensor, targets: Tensor) -> dict:
        sisdr = self.sisdr_loss(est_source, target_source)
        ce_loss = self.cross_entropy_loss(logits, targets)

        total_loss = sisdr + self.gamma * ce_loss["loss_ce"]

        return {"total_loss": total_loss, "sisdr_loss": sisdr, "cross_entropy_loss": ce_loss["loss_ce"]}
