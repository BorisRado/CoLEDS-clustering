import numpy as np

from src.testing.similarity import compute_similarity
from src.testing.correlation import compute_correlation
from src.testing.robustness import test_robustness
from src.testing.num_clusters import get_silhouette_scores
from src.data.utils import get_label_distribution, get_holdout_dataset
from src.clustering.clusterer import Clusterer



def test_all(cem, trainsets, valsets, n_classes, max_clusters=20, **kwargs):
    dataset_name = trainsets[0].dataset_name

    datasets = {
        "train": trainsets,
        "val": valsets,
    }

    gt_label_dist = np.vstack(
        [get_label_distribution(vs, n_classes=n_classes) for vs in valsets]
    )
    clusterer = Clusterer(cem, datasets)
    results = {
        "silhouette": get_silhouette_scores(clusterer.init_embeddings["train"], max_clusters=max_clusters),
        "correlation": compute_correlation(gt_label_dist, clusterer.init_embeddings["val"]),
        "similarity": compute_similarity(clusterer, gt_label_dist, max_clusters=max_clusters),
    }
    # if "femnist" not in dataset_name:
    #     holdout_set = get_holdout_dataset(dataset_name)
    #     results["robustness"] = test_robustness(trainsets, holdout_set, clusterer, n_classes, max_clusters=max_clusters, **kwargs)

    return results
