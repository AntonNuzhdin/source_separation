import torch.nn as nn
import torch

class IntraChunk(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(IntraChunk, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        fc_out = self.fc(rnn_out)
        ln_out = self.ln(fc_out)
        return ln_out

class InterChunk(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InterChunk, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        fc_out = self.fc(rnn_out)
        ln_out = self.ln(fc_out)
        return ln_out



