import pickle
import random
from functools import partial

import torch
import hydra
import pandas as pd
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.utils.stochasticity import set_seed
from src.data.utils import load_femnist_datasets, split_tensor_dataset
from src.flower.train import train_flower
from src.models.training_procedures import train_ce
from src.models.evaluation_procedures import get_all_client_accuracy
from src.utils.other import get_exp_folder, set_torch_flags


@hydra.main(version_base=None, config_path="../../conf", config_name="classification")
def run(cfg):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    home_folder = get_exp_folder()
    cfg.experiment.folder = str(home_folder)
    assert cfg.dataset.dataset_name == "femnist"

    set_seed(cfg.general.seed)
    datasets = load_femnist_datasets()

    # shuffle the datasets since femnist data loading is deterministic
    random.shuffle(datasets)

    n_holdout_clients = cfg.final_evaluation.n_holdout_clients
    n_train_clients = len(datasets) - n_holdout_clients
    holdout_datasets = datasets[n_train_clients:]
    print(f"Holdout clients: {len(holdout_datasets)}")

    # trainsets and valsets are used for training
    trainsets = datasets[:n_train_clients]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainsets, valsets = zip(*[split_tensor_dataset(ds, device) for ds in trainsets])

    model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape, n_classes=cfg.dataset.n_classes)

    train_fn = partial(train_ce, proximal_mu=cfg.train_config.proximal_mu)

    model, history = train_flower(
        model=model,
        client_fn_kwargs={
            "trainsets": trainsets,
            "valsets": valsets,
            "batch_size": cfg.train_config.batch_size,
            "train_fn": train_fn
        },
        optim_kwargs=OmegaConf.to_container(cfg.optimizer),
        n_rounds=cfg.train_config.num_epochs,
        experiment_folder=home_folder,
        strategy_kwargs=OmegaConf.to_container(cfg.strategy),
        model_save_name=f"model_weights.pth",
        seed=cfg.general.seed,
        model_return_strategy="best_accuracy",
        client_resources=OmegaConf.to_container(cfg.client_resources),
    )

    all_metrics = {}
    for k, v in history.metrics_distributed_fit.items():
        all_metrics[f"train_{k}"] = v
    for k, v in history.metrics_distributed.items():
        all_metrics[f"test_{k}"] = v
    with (home_folder / "metric_evolution.pkl").open("wb") as f:
        pickle.dump(all_metrics, f)

    # move tensors of existing TensorDatasets to device in-place (no new list / TensorDataset)
    for ds in holdout_datasets:
        ds.tensors = tuple(t.to(device) for t in ds.tensors)
    accuracy = get_all_client_accuracy(model, holdout_datasets)
    pd.DataFrame(accuracy).to_csv(home_folder / "ho_accuracy.csv", index=False)

    accuracy = get_all_client_accuracy(model, valsets)
    pd.DataFrame(accuracy).to_csv(home_folder / "val_accuracy.csv", index=False)

    accuracy = get_all_client_accuracy(model, trainsets)
    pd.DataFrame(accuracy).to_csv(home_folder / "train_accuracy.csv", index=False)


if __name__ == "__main__":
    set_torch_flags()
    run()
