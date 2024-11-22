import json
import os
import numpy as np
from glob import glob
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.model.lipreading.main import get_visual_embeddings_batch

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
        os.makedirs(os.path.join(self._data_dir, 'embeddings'), exist_ok=True)

        split_dir_audio = os.path.join(self._data_dir, 'audio', part)
        split_dir_mouths = Path(os.path.join(self._data_dir, 'mouths'))

        split_dir_audio_mix = os.path.join(split_dir_audio, 'mix')
        split_dir_audio_s1 = os.path.join(split_dir_audio, 's1')
        split_dir_audio_s2 = os.path.join(split_dir_audio, 's2')

        mix_paths = glob(os.path.join(split_dir_audio_mix, '*.wav'))
        s1_paths = glob(os.path.join(split_dir_audio_s1, '*.wav'))
        s2_paths = glob(os.path.join(split_dir_audio_s2, '*.wav'))

        if not (len(mix_paths) == len(s1_paths) == len(s2_paths)):
            raise ValueError("Inconsistent number of files in mix, s1, and s2 directories.")

        index = []
        mouth_paths_1 = []
        mouth_paths_2 = []
        for mix_path, s1_path, s2_path in zip(mix_paths, s1_paths, s2_paths):
            mix_path_ids = Path(mix_path).stem.split('_')
            mouth_paths_1.append(str(split_dir_mouths / (mix_path_ids[0] + '.npz')))
            mouth_paths_2.append(str(split_dir_mouths / (mix_path_ids[1] + '.npz')))

        num_samples = len(mix_paths)
        mouth_paths_1 = [str(split_dir_mouths / (Path(mix_path).stem.split('_')[0] + '.npz')) for mix_path in mix_paths]
        mouth_paths_2 = [str(split_dir_mouths / (Path(mix_path).stem.split('_')[1] + '.npz')) for mix_path in mix_paths]


        index = []
        embeddings_path = os.path.join(self._data_dir, os.path.join(self._data_dir, 'embeddings'))

        for i in tqdm(range(0, num_samples, 100), desc="Creating index..."):
            batch_end = min(i + 100, num_samples)
            batch_mouth_paths_1 = mouth_paths_1[i:batch_end]
            batch_mouth_paths_2 = mouth_paths_2[i:batch_end]
            batch_mix_paths = mix_paths[i:batch_end]
            batch_s1_paths = s1_paths[i:batch_end]
            batch_s2_paths = s2_paths[i:batch_end]

            embeddings_1 = get_visual_embeddings_batch(batch_mouth_paths_1)
            embeddings_2 = get_visual_embeddings_batch(batch_mouth_paths_2)

            for j, (mp1, mp2, mix_p, s1_p, s2_p) in enumerate(zip(batch_mouth_paths_1, batch_mouth_paths_2, batch_mix_paths, batch_s1_paths, batch_s2_paths)):
                mix_path_ids = Path(mix_p).stem.split('_')
                embedding_filename_1 = os.path.join(embeddings_path, f"{mix_path_ids[0]}_embedding.npy")
                embedding_filename_2 = os.path.join(embeddings_path, f"{mix_path_ids[1]}_embedding.npy")

                np.save(embedding_filename_1, embeddings_1[j])
                np.save(embedding_filename_2, embeddings_2[j])

                index.append({
                    'mix_path': mix_p,
                    's1_path': s1_p,
                    's2_path': s2_p,
                    's1_mouth_path': mp1,
                    's2_mouth_path': mp2,
                    'embed_s1': embedding_filename_1,
                    'embed_s2': embedding_filename_2,
                    'mix_audio_length': self._get_audio_length(mix_p),
                    's1_audio_length': self._get_audio_length(s1_p),
                    's2_audio_length': self._get_audio_length(s2_p),
                })
        return index

    def _get_audio_length(self, path):
        audio_info = torchaudio.info(str(path))
        return audio_info.num_frames / audio_info.sample_rate
