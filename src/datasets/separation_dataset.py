import json
import os
import shutil
from glob import glob
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class SourceSeparationDataset(BaseDataset):
    def __init__(self, part, data_dir, *args, **kwargs):
        self._data_dir = Path(data_dir)

        index = self._get_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []

        split_dir_audio = os.path.join(self._data_dir, 'audio', part)
        split_dir_mouths = Path(os.path.join(self._data_dir, 'mouths'))

        split_dir_audio_mix = os.path.join(split_dir_audio, 'mix')
        split_dir_audio_s1 = os.path.join(split_dir_audio, 's1')
        split_dir_audio_s2 = os.path.join(split_dir_audio, 's2')

        zip_mix_s1_s2 = zip(
            glob(os.path.join(split_dir_audio_mix, '*.wav')),
            glob(os.path.join(split_dir_audio_s1, '*.wav')),
            glob(os.path.join(split_dir_audio_s2, '*.wav'))
        )

        for mix_path, s1_path, s2_path in tqdm(zip_mix_s1_s2, desc="Creating index..."):
            mix_path_ids = Path(mix_path).name.split('.')[0].split('_')
            mix_path_ids = [x + '.npz' for x in mix_path_ids]
            index.append(
                {
                    'mix_path': mix_path,
                    's1_path': s1_path,
                    's2_path': s2_path,
                    's1_mouth_path': str(split_dir_mouths / mix_path_ids[0]),
                    's2_mouth_path': str(split_dir_mouths / mix_path_ids[1]),
                    'mix_audio_length': self._get_audio_length(mix_path),
                    's1_audio_length': self._get_audio_length(s1_path),
                    's2_audio_length': self._get_audio_length(s2_path),
                }
            )
        return index

    def _get_audio_length(self, path):
        audio_info = torchaudio.info(str(path))
        return audio_info.num_frames / audio_info.sample_rate
