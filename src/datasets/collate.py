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
    }

    for item in dataset_items:
        result_batch['mix_audio'].append(item['mix_audio'].squeeze(0))
        result_batch['s1_audio'].append(item['s1_audio'].squeeze(0))
        result_batch['s2_audio'].append(item['s2_audio'].squeeze(0))
        # result_batch['s1_mouth'].append(torch.tensor(item['s1_mouth']))
        # result_batch['s2_mouth'].append(torch.tensor(item['s2_mouth']))
        result_batch['mix_audio_length'].append(item['mix_audio_length'])
        result_batch['s1_audio_length'].append(item['s1_audio_length'])
        result_batch['s2_audio_length'].append(item['s2_audio_length'])
        result_batch['emb_s1'].append(item['emb_s1'])
        result_batch['emb_s2'].append(item['emb_s2'])

    def pad_sequences(sequences):
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    result_batch['mix_audio'] = pad_sequences(result_batch['mix_audio'])
    result_batch['s1_audio'] = pad_sequences(result_batch['s1_audio'])
    result_batch['s2_audio'] = pad_sequences(result_batch['s2_audio'])

    # result_batch['s1_mouth'] = torch.stack(result_batch['s1_mouth'])
    # result_batch['s2_mouth'] = torch.stack(result_batch['s2_mouth'])
    result_batch['mix_audio_length'] = torch.tensor(result_batch['mix_audio_length'])
    result_batch['s1_audio_length'] = torch.tensor(result_batch['s1_audio_length'])
    result_batch['s2_audio_length'] = torch.tensor(result_batch['s2_audio_length'])
    result_batch['emb_s1'] = torch.tensor(np.stack(result_batch['emb_s1'], axis=0))
    result_batch['emb_s2'] = torch.tensor(np.stack(result_batch['emb_s2'], axis=0))
    return result_batch
