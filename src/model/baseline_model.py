import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size,
                              out_channels=hidden_size,
                              kernel_size=kernel_size,
                              stride=stride)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)


class Masking(nn.Module):
    def __init__(self, hidden_size):
        super(Masking, self).__init__()

        self.mask_spk1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.mask_spk2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, x):
        mask1 = torch.sigmoid(self.mask_spk1(x))
        mask2 = torch.sigmoid(self.mask_spk2(x))
        return x * mask1, x * mask2


class Decoder(nn.Module):
    def __init__(self, hidden_size, kernel_size, stride):
        super(Decoder, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels=hidden_size,
                                         out_channels=1,
                                         kernel_size=kernel_size,
                                         stride=stride)

    def forward(self, x):
        return self.deconv(x)


class BaselineModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, kernel_size=16, stride=8):
        super(BaselineModel, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, kernel_size, stride)
        self.masking = Masking(hidden_size)
        self.decoder_spk1 = Decoder(hidden_size, kernel_size, stride)
        self.decoder_spk2 = Decoder(hidden_size, kernel_size, stride)

    def forward(self, mix_audio: torch.Tensor, **batch):
        mix_audio = mix_audio.unsqueeze(1)
        encoded_audio = self.encoder(mix_audio)

        masked_spk1, masked_spk2 = self.masking(encoded_audio)

        output_spk1 = self.decoder_spk1(masked_spk1).squeeze(1)
        output_spk2 = self.decoder_spk2(masked_spk2).squeeze(1)

        return {
            'output_spk1': output_spk1,
            'output_spk2': output_spk2
        }
