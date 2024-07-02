from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.utils import (
    get_label_distribution,
    get_dataset_target_indices,
    get_holdout_dataset
)
from src.testing.robustness import generate_similar_dataset
from src.models.evaluation_procedures import test


def get_holdout_accuracy(models_dict, clusterer, dataset_name, n_classes, trainsets, n_holdout_clients):
    holdout_dataset = get_holdout_dataset(dataset_name)
    label_distributions = [get_label_distribution(ds, n_classes=n_classes) for ds in trainsets]
    holdout_indices = get_dataset_target_indices(holdout_dataset)

    client_data = [
        (label_distributions[idx], trainsets[idx].dataset.applied_data_transforms)
        for idx in range(len(trainsets))
    ]
    out = {"dataset_size": [], "accuracy": [], "cluster_idx": []}
    for _ in tqdm(range(n_holdout_clients)):
        dataset = generate_similar_dataset(
            client_data,
            (0.1, 1.8),
            holdout_dataset,
            holdout_indices
        )
        dataloader = DataLoader(dataset, batch_size=32)

        pred_cluster = clusterer.predict_client_cluster(dataset)

        acc = test(models_dict[pred_cluster], dataloader)

        out["dataset_size"].append(len(dataset))
        out["accuracy"].append(acc)
        out["cluster_idx"].append(pred_cluster)

    return out


def get_femnist_holdout_accuracy(models_dict, clusterer, holdout_sets):
    out = {"dataset_size": [], "accuracy": [], "cluster_idx": []}

    for dataset in holdout_sets:
        pred_cluster = clusterer.predict_client_cluster(dataset)

        dataloader = DataLoader(dataset, batch_size=32)

        acc = test(models_dict[pred_cluster], dataloader)

        out["dataset_size"].append(len(dataset))
        out["accuracy"].append(acc)
        out["cluster_idx"].append(pred_cluster)
    return out
