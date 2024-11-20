import torch

def overlap_add(segments, segment_size, overlap, pad_length):
    step = segment_size - overlap
    batch_size, n_features, num_chunks, _ = segments.size()
    output_length = num_chunks * step + overlap
    output = torch.zeros(batch_size, n_features, output_length, device=segments.device)
    for i in range(num_chunks):
        start = i * step
        end = start + segment_size
        output[:, :, start:end] += segments[:, :, i, :]
    return output[:, :, :-pad_length] if pad_length > 0 else output
