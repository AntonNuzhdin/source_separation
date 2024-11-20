import torch
import torch.nn.functional as F

def segmentation(input_tensor, segment_size, overlap):
    step = segment_size - overlap
    batch_size, n_features, length = input_tensor.size()

    pad_length = (step - (length - segment_size) % step) % step
    input_tensor = F.pad(input_tensor, (0, pad_length))

    segments = input_tensor.unfold(-1, segment_size, step)
    return segments, pad_length
