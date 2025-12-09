import copy
import time

import torch
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf
from dotenv import load_dotenv

from src.data.utils import (
    get_dataloaders_with_replacement,
    get_datasets_from_cfg,
    to_pytorch_tensor_dataset
)
from src.utils.stochasticity import set_seed, TempRng
from src.cem.single_model_cem import SingleModelCEM
from src.models.training_procedures import train_contrastive
from src.models.helper import init_optimizer
from src.utils.evaluation import get_evaluation_fn
from src.utils.wandb import init_wandb, finish_wandb, run_exists_already
from src.utils.other import get_exp_folder, set_torch_flags, iterate_configs


"""This scripts train a model with the contrastive loss and tracks how the correlation between
the similarity of dataset embeddings and the similarity of the dataset labels changes over time.
"""

@hydra.main(version_base=None, config_path="../../conf", config_name="coleds")
def run_all(cfg):
    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainsets = to_pytorch_tensor_dataset(trainsets, n_classes=cfg.dataset.n_classes)
    valsets = to_pytorch_tensor_dataset(valsets, n_classes=cfg.dataset.n_classes)
    for single_cfg in iterate_configs(cfg, multirun_columns=[
        "train_config.num_client_updates",
        "train_config.temperature",
        "train_config.fraction_fit",
        "train_config.batch_size",
    ]):
        run(single_cfg, trainsets, valsets)


def run(cfg, trainsets, valsets):
    if run_exists_already(cfg):
        print("Run exists already. Returning...")
        return
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.general.seed)
    init_wandb(cfg)
    start_time = time.time()
    experiment_folder = get_exp_folder()
    cfg.experiment.folder = str(experiment_folder)

    with TempRng(cfg.general.seed):
        model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape)
    if torch.cuda.is_available():
        model = model.cuda()

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    # get all comparison values, i.e. before training and simple CEMs
    eval_fn = get_evaluation_fn(cfg, trainsets, valsets, experiment_folder)

    cem_fn = lambda mdl: SingleModelCEM(model=copy.deepcopy(mdl))
    best_correlation = eval_fn(cem=cem_fn(model), iter=-1)

    all_correlations = [best_correlation]
    rounds_without_improvement = 0

    trainloaders = get_dataloaders_with_replacement(
        trainsets,
        cfg.train_config.batch_size,
        cfg.dataset.horizontal_flipping,
    )

    optimizer = init_optimizer(
        model.parameters(),
        **OmegaConf.to_container(cfg.optimizer)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.general.max_epochs, eta_min=1e-8)

    for idx in range(cfg.general.max_epochs):
        print(f"Best correlation: {best_correlation}")
        train_contrastive(
            model=model,
            trainloaders=trainloaders,
            optimizer = optimizer,
            **OmegaConf.to_container(cfg.train_config),
        )
        tmp_corr = eval_fn(cem=cem_fn(model), iter=idx)
        print(f"Got correlation: {tmp_corr}")
        all_correlations.append(tmp_corr)

        if cfg.dataset.dataset_name == "synthetic":
            torch.save(model.state_dict(), experiment_folder / "cem.pth")
            break

        scheduler.step()
        if tmp_corr > best_correlation:
            best_correlation = tmp_corr
            rounds_without_improvement = 0
            if cfg.experiment.save_model:
                torch.save(model.state_dict(), experiment_folder / "cem.pth")
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= cfg.general.patience:
            print("Early exit...")
            break
    finish_wandb()
    print(f"Total training time: {time.time() - start_time:1f}")


if __name__ == "__main__":
    set_torch_flags()
    load_dotenv()
    run_all()
