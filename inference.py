import warnings

import hydra
import torch
from hydra.utils import instantiate
from pathlib import Path
import torchaudio
from src.datasets.custom_dataset import CustomDatasetInference
from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataloaders, _ = get_dataloaders(config, device, to_inference=True)

    model = instantiate(config.model).to(device)
    model.eval()
    print(model)

    # metrics = {"inference": []}
    # for metric_config in config.metrics.get("inference", []):
    #     metrics["inference"].append(
    #         instantiate(metric_config)
    #     )

    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        text_encoder=None,
        device=device,
        dataloaders=dataloaders,
        save_path=save_path,
        metrics=None,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    # for part in logs.keys():
    #     for key, value in logs[part].items():
    #         full_key = part + "_" + key
    #         print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
