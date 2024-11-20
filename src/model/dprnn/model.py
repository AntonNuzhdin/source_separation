import torch.nn as nn
from src.model.dprnn.dprnnblock import DPRNNBlock
from src.model.dprnn.segmentation import segmentation
from src.model.dprnn.overlap_add import overlap_add
import torch

class DPRNN(nn.Module):
    def __init__(self, encoder_dim, segment_size, overlap, hidden_dim, num_blocks, n_sources):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.segment_size = segment_size
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size=64, stride=32, bias=False)
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=64, stride=32, bias=False)
        self.num_blocks = num_blocks
        self.overlap = overlap
        self.n_sources = n_sources
        self.dprnn_blocks = nn.ModuleList([DPRNNBlock(encoder_dim, hidden_dim) for _ in range(num_blocks)])
        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(encoder_dim, encoder_dim * n_sources, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, mix_audio, **batch):
        mix_audio = mix_audio.unsqueeze(1)

        batch_size, _, signal_length = mix_audio.size()
        encoded_audio = self.encoder(mix_audio)
        segments, pad_length = segmentation(encoded_audio, self.segment_size, self.overlap)

        for block in self.dprnn_blocks:
            segments = block(segments)
        processed_audio = overlap_add(segments, self.segment_size, self.overlap, pad_length)
        processed_audio = processed_audio[:, :, :signal_length]
        mask = self.mask_generator(processed_audio)
        bs, _, seq = mask.size()
        mask = self.mask_generator(processed_audio)
        mask = mask.view(batch_size, self.n_sources, self.encoder_dim, seq)

        masked_audio = mask * processed_audio.unsqueeze(1)

        masked_audio = masked_audio.reshape(batch_size * self.n_sources, self.encoder_dim, -1)
        separated = self.decoder(masked_audio)
        separated = separated.reshape(batch_size, self.n_sources, -1)

        separated = separated[:, :, :signal_length]

        return {
            'speaker_1': separated[:, 0, :].squeeze(1),
            'speaker_2': separated[:, 1, :].squeeze(1)
        }

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

