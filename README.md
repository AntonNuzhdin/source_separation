# Project (Audio-Visual Source Separation)

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the custom realization of ConvTasNet and DPRNN source separation models.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/project_avss).

## Installation

Follow these steps to run the project:

## How To Use

0. Clone repository

```bash
git clone https://github.com/AntonNuzhdin/source_separation
cd source_separation
```
1. Create and activate env

```bash
conda create -n source_separation python=3.11.10

conda activate source_separation
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Dowload models weights

```bash
python download.py 
```

4. Run inference

To run the ConvTasNet (our best model)
```bash
python inference.py datasets.test.data_dir=<Path to wavs>
```

To run the additional DPRNN model
```bash
python inference.py datasets.test.data_dir=<Path to wavs> defaults.model=dprnn inferencer.from_pretrained="src/weights/dprnn_weights.pth"
```

You can see the text in the terminal: 'Saved predictions to:' {Path to predict}

Insert what is written instead 
{Path to predict}
 
5. Calculate metrics
   
{Path to wavs}: the path to the directory with ground truth

The structure should be like this:
```
{Path to wavs} / audio / ...

{Path to wavs}
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
```

```bash
python calc_metrics.py --estimated_path {Path to predict} --target_path {Path to wavs} --SISNRi --SISDRi --PESQ --STOI
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
