from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.utils.stochasticity import set_seed, TempRng
from src.data.utils import get_datasets_from_cfg
from src.clustering.clusterer import Clusterer
from src.flower.train import train_flower
from src.models.training_procedures import train_ce
from src.cem.helper import load_cem
from src.testing.accuracy import get_holdout_accuracy, get_femnist_holdout_accuracy


def train_cluster(cfg, trainsets, valsets, experiment_folder, cluster_idx, fraction_fit):
    print(f"Training cluster {cluster_idx} - START")

    with TempRng(cfg.general.seed):
        model = instantiate(cfg.model, input_shape=cfg.dataset.input_shape, n_classes=cfg.dataset.n_classes)

    set_seed(cfg.general.seed)
    train_fn = partial(train_ce, proximal_mu=cfg.train_config.proximal_mu)

    model = train_flower(
        model=model,
        client_fn_kwargs={
            "trainsets": trainsets,
            "valsets": valsets,
            "batch_size": cfg.train_config.batch_size,
            "train_fn": train_fn
        },
        optim_kwargs=OmegaConf.to_container(cfg.optimizer),
        n_rounds=150,
        experiment_folder=experiment_folder,
        strategy_kwargs={
            "fraction_fit": fraction_fit,
            "fraction_evaluate": 0.2
        },
        model_save_name=f"cluster_model_{cluster_idx}.pth",
        seed=cfg.general.seed
    )
    print(f"Training cluster {cluster_idx} - ENDED")

    return model


@hydra.main(version_base=None, config_path="../conf", config_name="train_clustering")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))

    folder = Path(cfg.folder)
    cem_config = OmegaConf.load(folder / "config.yaml")
    cem = load_cem(cem_config, folder)
    assert cem_config.dataset.dataset_name == cfg.dataset.dataset_name

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cem_config)
    shuffle_order = np.random.permutation(len(trainsets))
    trainsets = [trainsets[i] for i in shuffle_order]
    valsets = [valsets[i] for i in shuffle_order]

    del cem_config

    # predict the clusters
    if "femnist" in cfg.dataset.dataset_name:
        n_holdout_clients = cfg.final_evaluation.n_holdout_clients
        n_train_clients = len(trainsets) - n_holdout_clients
        holdout_datasets = trainsets[n_train_clients:]
        trainsets, valsets = trainsets[:n_train_clients], valsets[:n_train_clients]
        print(f"Holdout datasets: {len(holdout_datasets)}")

    datasets = {"train": trainsets}

    n_clusters = cfg.train_config.n_clusters
    clusterer = Clusterer(cem, datasets)
    clusters = clusterer.init_kmeans_model(n_clusters, partition="train")

    cluster_models = {}
    for cluster_idx in range(n_clusters):
        cluster_trainsets = [ds for ds, cls in zip(trainsets, clusters) if cls == cluster_idx]
        cluster_valsets = [ds for ds, cls in zip(valsets, clusters) if cls == cluster_idx]
        assert len(cluster_trainsets) == len(cluster_valsets) == (clusters == cluster_idx).sum()
        cluster_model = train_cluster(cfg, cluster_trainsets, cluster_valsets, folder, cluster_idx, fraction_fit=cfg.train_config.fraction_fit)
        cluster_models[cluster_idx] = cluster_model

    if "femnist" in cfg.dataset.dataset_name:
        accuracy = get_femnist_holdout_accuracy(cluster_models, clusterer, holdout_datasets)
        pd.DataFrame(accuracy).to_csv(folder / f"ho_accuracy_{n_clusters}.csv", index=False)

        accuracy = get_femnist_holdout_accuracy(cluster_models, clusterer, valsets)
        pd.DataFrame(accuracy).to_csv(folder / f"val_accuracy_{n_clusters}.csv", index=False)

    else:
        accuracy = get_holdout_accuracy(
            cluster_models,
            clusterer,
            cfg.dataset.dataset_name,
            cfg.dataset.n_classes,
            trainsets,
            cfg.final_evaluation.n_holdout_clients
        )
        pd.DataFrame(accuracy).to_csv(folder / f"ho_accuracy_{n_clusters}.csv", index=False)


if __name__ == "__main__":
    run()
