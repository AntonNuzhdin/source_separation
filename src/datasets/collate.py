import torch
import numpy as np


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {
        'mix_audio': [],
        's1_audio': [],
        's2_audio': [],
        's1_mouth': [],
        's2_mouth': [],
        'mix_audio_length': [],
        's1_audio_length': [],
        's2_audio_length': [],
        'emb_s1': [],
        'emb_s2': [],
        'mix_path': [],
        's1_path': [],
        's2_path': [],
    }

    for item in dataset_items:
        result_batch['mix_audio'].append(item['mix_audio'].squeeze(0))
        result_batch['mix_audio_length'].append(item['mix_audio_length'])
        result_batch['mix_path'].append(item['mix_path'])

        if item['s1_audio'] is not None:
            result_batch['s1_audio'].append(item['s1_audio'].squeeze(0))
            result_batch['s1_audio_length'].append(item['s1_audio_length'])
            result_batch['s1_path'].append(item['s1_path'])
        else:
            result_batch['s1_audio'].append(None)
            result_batch['s1_audio_length'].append(None)
            result_batch['s1_path'].append(None)

        if item['s2_audio'] is not None:
            result_batch['s2_audio'].append(item['s2_audio'].squeeze(0))
            result_batch['s2_audio_length'].append(item['s2_audio_length'])
            result_batch['s2_path'].append(item['s2_path'])
        else:
            result_batch['s2_audio'].append(None)
            result_batch['s2_audio_length'].append(None)
            result_batch['s2_path'].append(None)

    def pad_sequences(sequences):
        sequences = [seq for seq in sequences if seq is not None]
        if len(sequences) == 0:
            return None
        else:
            return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    result_batch['mix_audio'] = pad_sequences(result_batch['mix_audio'])
    result_batch['mix_audio_length'] = torch.tensor(result_batch['mix_audio_length'])

    if any(seq is not None for seq in result_batch['s1_audio']):
        result_batch['s1_audio'] = pad_sequences(result_batch['s1_audio'])
        result_batch['s1_audio_length'] = torch.tensor(
            [length for length in result_batch['s1_audio_length'] if length is not None]
        )
    else:
        result_batch['s1_audio'] = None
        result_batch['s1_audio_length'] = None

    if any(seq is not None for seq in result_batch['s2_audio']):
        result_batch['s2_audio'] = pad_sequences(result_batch['s2_audio'])
        result_batch['s2_audio_length'] = torch.tensor(
            [length for length in result_batch['s2_audio_length'] if length is not None]
        )
    else:
        result_batch['s2_audio'] = None
        result_batch['s2_audio_length'] = None

    return result_batch
