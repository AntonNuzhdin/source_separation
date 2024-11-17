import torch
import torch.nn as nn

class MultimodaFusion(nn.Module):
    def __init__(self, audio_dim, visual_dim, fused_dim):
        super(MultimodaFusion, self).__init__()
        self.fusion_proj = nn.Linear(audio_dim + visual_dim, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
        self.activation = nn.ReLU()

    def forward(self, audio_features, visual_features):
        audio_features = audio_features.permute(0, 2, 1)
        visual_features = visual_features.permute(0, 2, 1)
        combined_features = torch.cat([audio_features, visual_features], dim=-1)
        fused_features = self.fusion_proj(combined_features)
        fused_features = self.norm(fused_features)
        fused_features = self.activation(fused_features)
        return fused_features.permute(0, 2, 1)
