import os
import numpy as np
import torchaudio
import torch
from src.metrics.all_metrics import SISNRi, SISDRi, PESQ, STOI, SDRi
import argparse
from pathlib import Path


def load_audio(path, target_sr):
    if path is None:
        return None
    path = str(path)
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


def compute_metrics(args):
    metrics_list = [
        SISNRi(name="SISNRi"),
        SISDRi(name="SISDRi"),
        PESQ(fs=args.target_sr, mode="wb", name="PESQ"),
        STOI(fs=args.target_sr, extended=False, name="STOI"),
        SDRi(name="SDRi")
    ]
    metrics_results = {metric.name: [] for metric in metrics_list}

    estimated_root = Path(args.estimated_path)
    target_root = Path(args.target_path)

    s1_estimated = estimated_root / 'speaker_1'
    s2_estimated = estimated_root / 'speaker_2'

    s1_target = target_root / 'audio' / 's1'
    s2_target = target_root / 'audio' / 's2'
    mix_path = target_root / 'audio' / 'mix'

    required_dirs = [s1_estimated, s2_estimated, s1_target, s2_target, mix_path]
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"Directory {dir_path} does not exist")
            return

    filenames = set()
    for ext in ['wav', 'flac', 'mp3']:
        files = s1_estimated.glob(f'*.{ext}')
        filenames.update([f.stem for f in files])

    if not filenames:
        print(f"Don't find audio in {s1_estimated}")
        return

    for base_name in filenames:
        s1_est = None
        s2_est = None
        s1_true = None
        s2_true = None
        mix = None

        for ext in ['wav', 'flac', 'mp3']:
            est_path = s1_estimated / f'{base_name}.{ext}'
            if est_path.exists():
                s1_est = est_path
                break

        for ext in ['wav', 'flac', 'mp3']:
            est_path = s2_estimated / f'{base_name}.{ext}'
            if est_path.exists():
                s2_est = est_path
                break

        for ext in ['wav', 'flac', 'mp3']:
            true_path = s1_target / f'{base_name}.{ext}'
            if true_path.exists():
                s1_true = true_path
                break

        for ext in ['wav', 'flac', 'mp3']:
            true_path = s2_target / f'{base_name}.{ext}'
            if true_path.exists():
                s2_true = true_path
                break

        for ext in ['wav', 'flac', 'mp3']:
            mix_path_file = mix_path / f'{base_name}.{ext}'
            if mix_path_file.exists():
                mix = mix_path_file
                break

        if s1_est is None or s2_est is None or mix is None:
            print(f"Don't find {base_name}, skip")
            continue

        s1_est_audio = load_audio(s1_est, args.target_sr)
        s2_est_audio = load_audio(s2_est, args.target_sr)
        mix_audio = load_audio(mix, args.target_sr)

        s1_true_audio = None
        s2_true_audio = None

        if s1_true is not None:
            s1_true_audio = load_audio(s1_true, args.target_sr)
        if s2_true is not None:
            s2_true_audio = load_audio(s2_true, args.target_sr)

        for metric in metrics_list:
            try:
                result = metric(
                    metric=metric,
                    s1_audio=s1_true_audio,
                    s2_audio=s2_true_audio,
                    speaker_1=s1_est_audio,
                    speaker_2=s2_est_audio,
                    mix_audio=mix_audio
                )
                metrics_results[metric.name].append(result.item())
            except Exception as e:
                print(f"Erorr in {metric.name} for {base_name}: {e}")

    for metric_name, values in metrics_results.items():
        if values:
            avg_value = np.mean(values)
            print(f"{metric_name}: {avg_value:.4f}")
        else:
            print(f"No values computed for {metric_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics on inference")
    parser.add_argument("--estimated_path", required=True, type=str, help="Path to estimated")
    parser.add_argument("--target_path", required=True, type=str, help="Path to ground truth")
    parser.add_argument("--target_sr", default=16000, type=int, help="Sample rate")
    args = parser.parse_args()

    compute_metrics(args)
