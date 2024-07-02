from functools import partial

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg
from src.utils.evaluation import eval_fn
from src.utils.wandb import init_wandb, finish_wandb


@hydra.main(version_base=None, config_path="../conf", config_name="baseline")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    experiment_folder = init_wandb(cfg)
    trainsets, valsets = get_datasets_from_cfg(cfg)

    # get all comparison values, i.e. before training and simple CEMs
    eval_fn_ = partial(
        eval_fn,
        trainsets=trainsets,
        valsets=valsets,
        n_classes=cfg.dataset.n_classes,
        experiment_folder=experiment_folder
    )
    cem = instantiate(cfg.cem, n_classes=cfg.dataset.n_classes)

    eval_fn_(cem=cem, iter=0)
    finish_wandb()

if __name__ == "__main__":
    run()
