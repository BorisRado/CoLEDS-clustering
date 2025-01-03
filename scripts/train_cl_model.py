import copy

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
from src.utils.other import get_exp_folder, set_torch_flags


"""This scripts train a model with the contrastive loss and tracks how the correlation between
the similarity of dataset embeddings and the similarity of the dataset labels changes over time.
"""

@hydra.main(version_base=None, config_path="../conf", config_name="cl")
def run(cfg):

    # if run_exists_already(cfg):
    #     print("Run exists already. Returning...")
    #     return
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)
    set_torch_flags()
    experiment_folder = get_exp_folder()
    cfg.experiment_folder = experiment_folder

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainsets = to_pytorch_tensor_dataset(trainsets)
    valsets = to_pytorch_tensor_dataset(valsets)

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

    trainloaders = get_dataloaders_with_replacement(
        trainsets,
        cfg.train_config.batch_size
    )

    optimizer = init_optimizer(
        model.parameters(),
        **OmegaConf.to_container(cfg.optimizer)
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.85)

    for idx in range(20):
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
            torch.save(model.state_dict(), experiment_folder / "cem.pth")

        if len(all_correlations) > 5 and all_correlations[-1] < min(all_correlations[-6:-1]):
            print("Early exit...")
            break
    finish_wandb()


if __name__ == "__main__":
    load_dotenv()
    run()
