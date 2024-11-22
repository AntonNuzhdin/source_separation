import numpy as np
import torchaudio
from src.metrics.all_metrics import SISNRi, SISDRi, PESQ, STOI
import argparse
from pathlib import Path
from tqdm import tqdm


def find_audio_file(base_name, directory, extensions=('wav', 'flac', 'mp3')):
    directory = Path(directory)
    for ext in extensions:
        matching_files = list(directory.glob(f"{base_name}.{ext}"))
        if matching_files:
            return matching_files[0]
    return None


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
    metrics_list = []

    if args.SISNRi:
        metrics_list.append(SISNRi(name="SISNRi"))
    if args.SISDRi:
        metrics_list.append(SISDRi(name="SISDRi"))
    if args.PESQ:
        metrics_list.append(PESQ(fs=args.target_sr, mode="wb", name="PESQ"))
    if args.STOI:
        metrics_list.append(STOI(fs=args.target_sr, extended=False, name="STOI"))

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
        assert Path(dir_path).exists(), f"Directory {dir_path} does not exist"

    filenames = set()
    for ext in ['wav', 'flac', 'mp3']:
        files = s1_estimated.glob(f'*.{ext}')
        filenames.update([f.stem for f in files])

    assert len(filenames) != 0, f"Don't find audio in {s1_estimated}"

    for base_name in tqdm(filenames):
        s1_est = find_audio_file(base_name, s1_estimated)
        s2_est = find_audio_file(base_name, s2_estimated)
        s1_true = find_audio_file(base_name, s1_target)
        s2_true = find_audio_file(base_name, s2_target)
        mix = find_audio_file(base_name, mix_path)

        if s1_est is None or s2_est is None:
            print(f"Don't find {base_name}, skip")
            continue

        s1_est_audio = load_audio(s1_est, args.target_sr)
        s2_est_audio = load_audio(s2_est, args.target_sr)
        mix_audio = load_audio(mix, args.target_sr)

        s1_true_audio = load_audio(s1_true, args.target_sr)
        s2_true_audio = load_audio(s2_true, args.target_sr)

        if s1_true_audio is None or s2_true_audio is None:
            continue

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
                print(f"Error in {metric.name} for {base_name}: {e}")

    for metric_name, values in metrics_results.items():
        if values:
            print(f"{metric_name}:", np.mean([v for v in values if v != 0]))
        else:
            print(f"{metric_name}: No values calculated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics on inference")
    parser.add_argument("--estimated_path", required=True, type=str, help="Path to estimated")
    parser.add_argument("--target_path", required=True, type=str, help="Path to ground truth")
    parser.add_argument("--target_sr", default=16000, type=int, help="Sample rate")

    parser.add_argument("--SISNRi", default=True, action="store_true", help="Enable SISNRi metric")
    parser.add_argument("--SISDRi", action="store_true", help="Enable SISDRi metric")
    parser.add_argument("--PESQ", action="store_true", help="Enable PESQ metric")
    parser.add_argument("--STOI", action="store_true", help="Enable STOI metric")

    args = parser.parse_args()

    compute_metrics(args)
