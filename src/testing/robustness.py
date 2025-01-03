import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.data.utils import (
    get_dataset_target_indices,
    sample_subset_given_distribution,
    get_label_distribution,
)


class _TmpDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        self.transform = T.Compose([
            t for t in self.transform.transforms if not isinstance(t, T.ToTensor)
        ])

    def __getitem__(self, index):
        img, label = self.subset[index]
        img = self.transform(img)
        batch = (img, label )
        return batch

    def __len__(self):
        return len(self.subset)


def generate_similar_dataset(cluster_client_data, scaling, holdout_set, holdout_indices):
    ref_idx = np.random.choice(len(cluster_client_data))
    ref_dist, ref_transforms = cluster_client_data[ref_idx]

    # obtain a similar distribution
    while True:
        tmp_dist = ref_dist * np.random.uniform(low=scaling[0], high=scaling[1])
        tmp_dist = np.rint(tmp_dist).astype(np.int32)  # round to integers
        if tmp_dist.sum() > 8:
            break
    # get the target dataset
    subset = sample_subset_given_distribution(holdout_set, holdout_indices, tmp_dist)
    subset = _TmpDataset(subset, ref_transforms)
    return subset



def _compute_cluster_robustness(
    clusterer,
    cluster_idx,
    holdout_indices,
    holdout_set,
    cluster_client_data,
    n_iterations=50,
    scaling=(0.2, 1.4)
):
    count = 0

    for _ in range(n_iterations):
        subset = generate_similar_dataset(cluster_client_data, scaling, holdout_set, holdout_indices)
        pred_cluster = clusterer.predict_client_cluster(subset)
        count += int(pred_cluster == cluster_idx)
    return count / n_iterations


def get_cluster_robustness(
    n_clusters,
    clusterer,
    label_distributions,
    trainsets,
    holdout_indices,
    holdout_set,
    **kwargs
):
    client_clusters = clusterer.init_kmeans_model(n_clusters=n_clusters)

    out = []
    print(f"Robustness for {n_clusters}")
    for cluster_idx in range(n_clusters):
        cluster_data = [
            (label_distributions[idx], trainsets[idx].applied_data_transforms)
            for idx, c in enumerate(client_clusters) if c == cluster_idx
        ]
        cluster_robustness = _compute_cluster_robustness(
            clusterer=clusterer,
            cluster_idx=cluster_idx,
            holdout_indices=holdout_indices,
            holdout_set=holdout_set,
            cluster_client_data=cluster_data,
            **kwargs
        )
        out.append({
            "n_clusters": int(n_clusters),
            "cluster_idx": int(cluster_idx),
            "n_cluster_clients": len(cluster_data),
            "robustness": float(cluster_robustness)
        })
    print(f"Robustness for {n_clusters} completed")
    return out


def _flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def test_robustness(trainsets, holdout_set, clusterer, n_classes, max_clusters=30, **kwargs):
    label_distributions = [get_label_distribution(ds, n_classes=n_classes) for ds in trainsets]

    holdout_indices = get_dataset_target_indices(holdout_set)


    results = [get_cluster_robustness(
        idx,
        clusterer=clusterer,
        label_distributions=label_distributions,
        trainsets=trainsets,
        holdout_indices=holdout_indices,
        holdout_set=holdout_set,
        **kwargs
    ) for idx in range(1, max_clusters)]

    return _flatten(results)
