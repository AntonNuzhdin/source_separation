import torch
import torch.nn as nn
from asteroid.models import ConvTasNet


class BaselineModel(nn.Module):
    def __init__(self, num_sources=2):
        super(BaselineModel, self).__init__()
        self.model = ConvTasNet(n_src=num_sources)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture (torch.Tensor): Tensor containing the mixture audio of shape (batch, time).

        Returns:
            torch.Tensor: Tensor containing the separated sources of shape (batch, num_sources, time).
        """
        with torch.no_grad():
            separated_sources = self.model(mixture)
        return separated_sources
