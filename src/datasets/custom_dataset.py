import os
from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from tqdm import tqdm
import torchaudio

class CustomDatasetInference(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        self._data_dir = Path(data_dir)
        self.index = self._create_index()
        self.sample_rate = 16_000
        super().__init__(self.index, *args, **kwargs)

    def _create_index(self):
        index = []
        mix_dir = self._data_dir / "audio" / "mix"

        if mix_dir.exists():
            # print(str(mix_dir))
            mix_files = sorted([f for f in mix_dir.glob("*.*") if f.suffix.lower() in ['.wav', '.flac', '.mp3']])
            for file_m in tqdm(mix_files, desc = 'Create dataset...'):
                path_mix = str(file_m)
                path_wo_extension = file_m.stem
                parts = path_wo_extension.split('_')

                speaker1, speaker2 = parts

                s1_p = self._find_audio_file(self._data_dir / "audio" / "s1", f"{speaker1}_{speaker2}")
                s2_p = self._find_audio_file(self._data_dir / "audio" / "s2", f"{speaker1}_{speaker2}")

                index.append({
                "mix_path": path_mix,
                "s1_path": str(s1_p) if s1_p else None,
                "s2_path": str(s2_p) if s2_p else None,
                "mix_audio_length": self._get_audio_length(file_m),
                "s1_audio_length": self._get_audio_length(s1_p) if s1_p else None,
                "s2_audio_length": self._get_audio_length(s2_p) if s2_p else None,
            })

            return index
        else:
            raise ValueError(f"{str(mix_dir)} directory not exist")

    def _get_audio_length(self, path):
        audio_info = torchaudio.info(str(path))
        return audio_info.num_frames / audio_info.sample_rate

    def _find_audio_file(self, dir, file_name):
        for ext in ["wav", "flac", "mp3"]:
            file_path = dir / f"{file_name}.{ext}"
            if file_path.exists():
                return file_path
        return None

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        else:
            audio_tensor = audio_tensor[0:1, :]
        if sr != self.sample_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.sample_rate)
        return audio_tensor


    def __getitem__(self, ind):
        data_dict = self.index[ind]

        mix_path = data_dict['mix_path']
        s1_path = data_dict.get('s1_path')
        s2_path = data_dict.get('s2_path')
        mix_audio_length = data_dict['mix_audio_length']
        s1_audio_length = data_dict.get('s1_audio_length')
        s2_audio_length = data_dict.get('s2_audio_length')

        mix_audio = self.load_audio(mix_path)

        if s1_path:
            s1_audio = self.load_audio(s1_path)
        else:
            s1_audio = None

        if s2_path:
            s2_audio = self.load_audio(s2_path)
        else:
            s2_audio = None

        instance_data = {
            'mix_path': mix_path,
            's1_path': s1_path,
            's2_path': s2_path,
            'mix_audio_length': mix_audio_length,
            's1_audio_length': s1_audio_length,
            's2_audio_length': s2_audio_length,
            'mix_audio': mix_audio,
            's1_audio': s1_audio,
            's2_audio': s2_audio,
        }

        return instance_data
