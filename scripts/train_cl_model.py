import copy

import torch
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.data.utils import get_dataloaders_with_replacement, get_datasets_from_cfg
from src.utils.stochasticity import set_seed, TempRng
from src.cem.single_model_cem import SingleModelCEM
from src.models.training_procedures import train_contrastive
from src.models.helper import init_optimizer
from src.utils.evaluation import get_evaluation_fn
from src.utils.wandb import init_wandb, finish_wandb


@hydra.main(version_base=None, config_path="../conf", config_name="cl")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    experiment_folder = init_wandb(cfg)

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)

    with TempRng(cfg.general.seed):
        model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape)

    print("number of parameters: ", sum(p.numel() for p in model.parameters()))

    # get all comparison values, i.e. before training and simple CEMs
    eval_fn = get_evaluation_fn(cfg, trainsets, valsets, experiment_folder)

    cem_fn = lambda mdl: SingleModelCEM(copy.deepcopy(mdl))
    best_correlation = eval_fn(cem=cem_fn(model), iter=-1)

    trainloaders = get_dataloaders_with_replacement(
        trainsets,
        cfg.train_config.batch_size
    )

    optimizer = init_optimizer(
        model.parameters(),
        **OmegaConf.to_container(cfg.optimizer)
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
    idx = 0
    while True:
        print(f"Best correlation: {best_correlation}")
        train_contrastive(
            model=model,
            n_iterations=cfg.train_config.n_iterations,
            temperature=cfg.train_config.temperature,
            dataloaders=trainloaders,
            fraction_fit=cfg.train_config.fraction_fit,
            optimizer = optimizer
        )
        tmp_corr = eval_fn(cem=cem_fn(model), iter=idx)
        print(f"Got correlation: {tmp_corr}")
        if cfg.dataset.dataset_name == "synthetic":
            torch.save(model.state_dict(), experiment_folder / "cem.pth")
            break

        if tmp_corr > best_correlation and idx <= 5:
            best_correlation = tmp_corr
            scheduler.step()
            idx += 1
            torch.save(model.state_dict(), experiment_folder / "cem.pth")
        else:
            break

    finish_wandb()


if __name__ == "__main__":
    run()
