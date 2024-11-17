import torch


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
        's2_audio_length': []
    }

    for item in dataset_items:
        result_batch['mix_audio'].append(item['mix_audio'].squeeze(0))
        result_batch['s1_audio'].append(item['s1_audio'].squeeze(0))
        result_batch['s2_audio'].append(item['s2_audio'].squeeze(0))
        result_batch['s1_mouth'].append(item['s1_mouth'])
        result_batch['s2_mouth'].append(item['s2_mouth'])
        result_batch['mix_audio_length'].append(item['mix_audio_length'])
        result_batch['s1_audio_length'].append(item['s1_audio_length'])
        result_batch['s2_audio_length'].append(item['s2_audio_length'])

    def pad_sequences(sequences):
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    result_batch['mix_audio'] = pad_sequences(result_batch['mix_audio'])
    result_batch['s1_audio'] = pad_sequences(result_batch['s1_audio'])
    result_batch['s2_audio'] = pad_sequences(result_batch['s2_audio'])

    result_batch['s1_mouth'] = torch.stack(result_batch['s1_mouth'])
    result_batch['s2_mouth'] = torch.stack(result_batch['s2_mouth'])
    result_batch['mix_audio_length'] = torch.tensor(result_batch['mix_audio_length'])
    result_batch['s1_audio_length'] = torch.tensor(result_batch['s1_audio_length'])
    result_batch['s2_audio_length'] = torch.tensor(result_batch['s2_audio_length'])

    return result_batch
