import os
import torch
import torch.nn as nn
import numpy as np

from src.model.lipreading.lipreading.utils import load_json
from src.model.lipreading.lipreading.model import Lipreading
from src.model.lipreading.lipreading.dataloaders import get_preprocessing_pipelines


PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PATH, 'configs', 'lrw_resnet18_dctcn.json')


def load_model(load_path, model, optimizer = None, allow_size_mismatch = False):
    assert os.path.isfile( load_path ), "Error when loading the model, provided path not found: {}".format( load_path )
    checkpoint = torch.load(load_path)
    loaded_state_dict = checkpoint['model_state_dict']

    if allow_size_mismatch:
        loaded_sizes = { k: v.shape for k,v in loaded_state_dict.items() }
        model_state_dict = model.state_dict()
        model_sizes = { k: v.shape for k,v in model_state_dict.items() }
        mismatched_params = []
        for k in loaded_sizes:
            if loaded_sizes[k] != model_sizes[k]:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]

    model.load_state_dict(loaded_state_dict, strict = not allow_size_mismatch)
    print(f'Loaded lipreading model from {load_path}')
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch_idx'], checkpoint
    return model


def load_model_from_json(config_path):
    assert config_path.endswith('.json') and os.path.isfile(config_path), \
        f"'.json' config path does not exist. Path input: {config_path}"

    args_loaded = load_json(config_path)
    backbone_type = args_loaded['backbone_type']
    width_mult = args_loaded['width_mult']
    relu_type = args_loaded['relu_type']
    use_boundary = args_loaded.get("use_boundary", False)

    tcn_options = {
        'num_layers': args_loaded.get('tcn_num_layers', 4),
        'kernel_size': args_loaded.get('tcn_kernel_size', [3]),
        'dropout': args_loaded.get('tcn_dropout', 0.2),
        'dwpw': args_loaded.get('tcn_dwpw', False),
        'width_mult': args_loaded.get('tcn_width_mult', 1),
    } if args_loaded.get('tcn_num_layers', '') else {}

    densetcn_options = {
        'block_config': args_loaded.get('densetcn_block_config', [3, 4, 6, 3]),
        'growth_rate_set': args_loaded.get('densetcn_growth_rate_set', [32]),
        'reduced_size': args_loaded.get('densetcn_reduced_size', 256),
        'kernel_size_set': args_loaded.get('densetcn_kernel_size_set', [3]),
        'dilation_size_set': args_loaded.get('densetcn_dilation_size_set', [1]),
        'squeeze_excitation': args_loaded.get('densetcn_se', False),
        'dropout': args_loaded.get('densetcn_dropout', 0.2),
    } if args_loaded.get('densetcn_block_config', '') else {}

    model = Lipreading(
        modality='video',
        num_classes=2,
        tcn_options=tcn_options,
        densetcn_options=densetcn_options,
        backbone_type=backbone_type,
        relu_type=relu_type,
        width_mult=width_mult,
        use_boundary=use_boundary,
        extract_feats=True
    ).cuda()
    return model

def extract_features_batch_model(model, mouth_patches_batch):
    model.eval()
    preprocessing_func = get_preprocessing_pipelines(modality='video')['test']
    processed_data_list = [preprocessing_func(mouth_patch) for mouth_patch in mouth_patches_batch]

    lengths = [data.shape[0] for data in processed_data_list]

    max_length = max(lengths)
    padded_data = []
    for data in processed_data_list:
        pad_width = ((0, max_length - data.shape[0]), (0, 0), (0, 0))
        padded_data.append(np.pad(data, pad_width, mode='constant'))

    batch_data = np.stack(padded_data, axis=0)

    with torch.no_grad():
        features = model(torch.FloatTensor(batch_data)[:, None, :, :, :].cuda(), lengths=lengths)

    return features.detach()


def get_visual_embeddings(mouth_patches_batch):
    model = load_model_from_json(CONFIG_PATH)
    model = load_model(os.path.join(PATH, 'weights', 'lrw_resnet18_dctcn_video_boundary.pth'), model, allow_size_mismatch=True)
    features = extract_features_batch_model(model, mouth_patches_batch)
    return features


# def extract_features_one(model, mouth_patch_path):
#     model.eval()
#     preprocessing_func = get_preprocessing_pipelines(modality='video')['test']
#     data = preprocessing_func(np.load(mouth_patch_path)['data'])  # data: TxHxW
#     with torch.no_grad():
#         features = model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])
#     return features.cpu().detach().numpy()


# def get_visual_embeddings_one(mouth_patch_path):
#     model = load_model_from_json(CONFIG_PATH)
#     model = load_model(os.path.join(PATH, 'weights', 'lrw_resnet18_dctcn_video_boundary.pth'), model, allow_size_mismatch=True)
#     features = extract_features_one(model, mouth_patch_path)
#     return features


def extract_features_batch(model, mouth_patch_paths):
  model.eval()
  preprocessing_func = get_preprocessing_pipelines(modality='video')['test']

  data_batch = []
  lengths = []
  for path in mouth_patch_paths:
    data = np.load(path)['data'] # data: TxHxW
    data_batch.append(data)
    lengths.append(data.shape[0])


  data_tensor = torch.FloatTensor(np.stack(data_batch)).cuda()
  data_tensor = data_tensor.unsqueeze(1)

  with torch.no_grad():
    features = model(data_tensor, lengths=lengths)

  return features.cpu().detach().numpy()


def get_visual_embeddings_batch(mouth_patch_paths):
  model = load_model_from_json(CONFIG_PATH)
  model = load_model(os.path.join(PATH, 'weights', 'lrw_resnet18_dctcn_video_boundary.pth'), model, allow_size_mismatch=True)
  model.cuda()

  features = extract_features_batch(model, mouth_patch_paths)
  return features
