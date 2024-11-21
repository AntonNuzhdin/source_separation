# Project (Audio-Visual Source Separation)

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving Source Separation task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/project_avss).

## Installation

Follow these steps to install the project:

## How To Use

0. Clone repository

```bash
git clone https://github.com/AntonNuzhdin/source_separation
```
1. Move to folder

```bash
cd source_separation
```

2. Create and activate env

```bash
conda create -n tmp python=3.11.10

conda activate tmp
```

3. Install requirements

```bash
pip install -r requirements.txt
```

4. Dowload model weights

```bash
python download.py 
```

5. Run inference

```bash
python inference.py datasets.test.data_dir=<Path to wavs>
```

You can see the text in the terminal: 'Saved predictions to:' <Path to predict> 

Paste into next paragraph

6. Calculate metrics

```bash
python calc_metrics.py     --estimated_path <Path to predict>    --target_path <Path to wavs>    --target_sr 16000
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
