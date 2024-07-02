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

    with TempRng(cfg.general.seed):
        model = instantiate(
            cfg.model,
            input_shape=cfg.dataset.input_shape,
            n_classes=cfg.dataset.n_classes
        )

    if cfg.dataset.dataset_name != "synthetic":
        eval_fn_ = partial(
            eval_fn,
            trainsets=trainsets,
            valsets=valsets,
            n_classes=cfg.dataset.n_classes,
            experiment_folder=experiment_folder
        )
    else:
        eval_fn_ = lambda *args, **kwargs: 1

    cem_fn = instantiate(cfg.cem, _partial_=True)

    cem = cem_fn(model=model)
    best_correlation = -1
    # best_correlation = eval_fn_(cem=cem, iter=-1)


    train_fn = partial(train_supervised_autoencoder, ae_weight=cfg.train_config.ae_weight, proximal_mu=cfg.train_config.proximal_mu)
    idx = 0
    while True:
        print(f"Best correlation: {best_correlation}")

        model = train_flower(
            model=model,
            trainsets=trainsets,
            valsets=valsets,
            batch_size=cfg.train_config.batch_size,
            optim_kwargs=OmegaConf.to_container(cfg.optimizer),
            n_rounds=cfg.train_config.cem_evaluation_freq,
            experiment_folder=experiment_folder,
            train_fn=train_fn,
            fraction_fit=cfg.train_config.fraction_fit
        )
        cem = cem_fn(model=model)
        tmp_corr = eval_fn_(cem=cem, iter=idx)

        if cfg.dataset.dataset_name == "synthetic":
            torch.save(model.state_dict(), experiment_folder / "cem.pth")
            break

        print(f"Got correlation: {tmp_corr}")
        if tmp_corr > best_correlation:
            best_correlation = tmp_corr
            idx += 1
            torch.save(model.state_dict(), experiment_folder / "cem.pth")

        else:
            break
    finish_wandb()


if __name__ == "__main__":
    run()
