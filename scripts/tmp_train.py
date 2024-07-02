from functools import partial

import torch
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg
from src.utils.evaluation import eval_fn
from src.utils.stochasticity import set_seed, TempRng
from src.utils.wandb import init_wandb, finish_wandb
from src.flower.train import train_flower
from src.models.training_procedures import train_supervised_autoencoder


@hydra.main(version_base=None, config_path="../conf", config_name="es")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    experiment_folder = init_wandb(cfg)

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)
    assert len(trainsets) == 1
    dataloader = torch.utils.data.DataLoader(trainsets[0])

    with TempRng(cfg.general.seed):
        model = instantiate(
            cfg.model,
            input_shape=cfg.dataset.input_shape,
            n_classes=cfg.dataset.n_classes
        )

    for _ in range(2):
        train_supervised_autoencoder(
            model,
            dataloader,
            cfg.train_config.ae_weight,
            **OmegaConf.to_container(cfg.optimizer)
        )
    torch.save(model.state_dict(), experiment_folder / "fl_model.pth")

    finish_wandb()


if __name__ == "__main__":
    run()
