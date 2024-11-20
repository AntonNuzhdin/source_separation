import torch.nn as nn
from src.model.dprnn.utils import IntraChunk, InterChunk

import torch

class DPRNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DPRNNBlock, self).__init__()
        self.intra_chunk = IntraChunk(input_dim, hidden_dim)
        self.inter_chunk = InterChunk(input_dim, hidden_dim)

    def forward(self, x):
        batch_size, n_features, num_chunks, segment_size = x.size()

        intra_input = x.permute(0, 3, 2, 1).contiguous()
        intra_input = intra_input.view(batch_size * segment_size, num_chunks, n_features)
        intra_output = self.intra_chunk(intra_input)
        intra_output = intra_output.view(batch_size, segment_size, num_chunks, n_features).permute(0, 3, 2, 1)
        intra_output = x + intra_output

        inter_input = intra_output.permute(0, 2, 3, 1).contiguous()
        inter_input = inter_input.view(batch_size * num_chunks, segment_size, n_features)
        inter_output = self.inter_chunk(inter_input)
        inter_output = inter_output.view(batch_size, num_chunks, segment_size, n_features).permute(0, 3, 1, 2)
        inter_output = intra_output + inter_output

        return inter_output



