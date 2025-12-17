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
from src.clustering.clusterer import Clusterer
from src.flower.train import train_flower
from src.models.training_procedures import train_ce
from src.profiler.helper import load_profiler
from src.models.evaluation_procedures import get_clustering_accuracy
from src.utils.other import set_torch_flags


def train_cluster(cfg, trainsets, valsets, experiment_folder, cluster_idx):
    print(f"Training cluster {cluster_idx} - START")

    model = instantiate(cfg.model, input_shape=(1, 28, 28), n_classes=62) # ASSUME FEMNIST

    set_seed(cfg.general.seed)
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
        experiment_folder=experiment_folder,
        strategy_kwargs=OmegaConf.to_container(cfg.strategy),
        model_save_name=f"cluster_model_{cluster_idx}.pth",
        seed=cfg.general.seed,
        model_return_strategy="best_accuracy"
    )

    print(f"Training cluster {cluster_idx} - ENDED")

    return model. history


@hydra.main(version_base=None, config_path="../../conf", config_name="train_clustering")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))

    profiler, home_folder = load_profiler(cfg)

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

    n_clusters = cfg.train_config.n_clusters
    clusterer = Clusterer(profiler, {"train": trainsets})
    clusters = clusterer.init_kmeans_model(n_clusters)

    cluster_models = {}
    for cluster_idx in range(n_clusters):
        cluster_trainsets = [ds for ds, cls in zip(trainsets, clusters) if cls == cluster_idx]
        print(f"Training cluster {cluster_idx} with {len(cluster_trainsets)} training clients")
        cluster_valsets = [ds for ds, cls in zip(valsets, clusters) if cls == cluster_idx]
        assert len(cluster_trainsets) == len(cluster_valsets) == (clusters == cluster_idx).sum()
        cluster_model, history = train_cluster(
            cfg,
            cluster_trainsets,
            cluster_valsets,
            home_folder,
            cluster_idx,
        )
        cluster_models[cluster_idx] = cluster_model
        all_metrics = {}
        for k, v in history.metrics_distributed_fit.items():
            all_metrics[f"train_{k}"] = v
        for k, v in history.metrics_distributed.items():
            all_metrics[f"test_{k}"] = v
        with (home_folder / f"metric_evolution_{cluster_idx}_oo{n_clusters}.pkl").open("wb") as f:
            pickle.dump(all_metrics, f)

    # move tensors of existing TensorDatasets to device in-place (no new list / TensorDataset)
    for ds in holdout_datasets:
        ds.tensors = tuple(t.to(device) for t in ds.tensors)
    accuracy = get_clustering_accuracy(cluster_models, clusterer, holdout_datasets)
    pd.DataFrame(accuracy).to_csv(home_folder / f"ho_accuracy_{n_clusters}.csv", index=False)

    accuracy = get_clustering_accuracy(cluster_models, clusterer, valsets)
    pd.DataFrame(accuracy).to_csv(home_folder / f"val_accuracy_{n_clusters}.csv", index=False)

    accuracy = get_clustering_accuracy(cluster_models, clusterer, trainsets)
    pd.DataFrame(accuracy).to_csv(home_folder / f"train_accuracy_{n_clusters}.csv", index=False)


if __name__ == "__main__":
    set_torch_flags()
    run()
