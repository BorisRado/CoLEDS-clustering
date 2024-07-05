from functools import partial

import torch
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg
from src.utils.evaluation import get_evaluation_fn
from src.utils.stochasticity import set_seed, TempRng
from src.utils.wandb import init_wandb, finish_wandb
from src.flower.train import train_flower
from src.models.training_procedures import train_supervised_autoencoder
from flwr.common import log

@hydra.main(version_base=None, config_path="../conf", config_name="es")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    experiment_folder = init_wandb(cfg)

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)

    with TempRng(cfg.general.seed):
        model = instantiate(
            cfg.model,
            input_shape=cfg.dataset.input_shape,
            n_classes=cfg.dataset.n_classes
        )

    eval_fn= get_evaluation_fn(cfg, trainsets, valsets, experiment_folder)
    cem_fn = instantiate(cfg.cem, _partial_=True)

    # best_correlation = eval_fn(cem=cem_fn(model=model), iter=-1)
    best_correlation = -1

    train_fn = partial(
        train_supervised_autoencoder,
        ae_weight=cfg.train_config.ae_weight,
        proximal_mu=cfg.train_config.proximal_mu
    )

    idx = 0
    while True:
        print(f"Best correlation: {best_correlation}")

        model = train_flower(
            model=model,
            client_fn_kwargs={
                "trainsets": trainsets,
                "valsets": valsets,
                "batch_size": cfg.train_config.batch_size,
                "train_fn": train_fn,
            },
            optim_kwargs=OmegaConf.to_container(cfg.optimizer),
            n_rounds=cfg.train_config.cem_evaluation_freq,
            experiment_folder=experiment_folder,
            strategy_kwargs={
                "fraction_fit": cfg.train_config.fraction_fit,
                "fraction_evaluate": 0.25  # just for debugging...
            },
            seed=cfg.general.seed
        )
        if cfg.dataset.dataset_name == "synthetic":
            break

        tmp_corr = eval_fn(cem=cem_fn(model=model), iter=idx)
        print(f"Got correlation: {tmp_corr}")
        if tmp_corr > best_correlation and idx <= 5:
            best_correlation = tmp_corr
            idx += 1
        else:
            break

    finish_wandb()


if __name__ == "__main__":
    run()
