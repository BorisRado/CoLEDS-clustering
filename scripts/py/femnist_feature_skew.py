import os
import numpy as np
import torch

import hydra
from hydra.utils import instantiate
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.utils.other import set_torch_flags
from scripts.py.train_coleds import run as train_coleds
from src.data.utils import load_femnist_datasets
from src.profiler.single_model_profiler import SingleModelProfiler
from scipy.spatial.distance import cdist


@hydra.main(version_base=None, config_path="../../conf", config_name="coleds")
def run_all(cfg):
    assert cfg.dataset.dataset_name == "femnist"
    cfg.experiment.save_model = True
    target_label = cfg.target_label
    datasets = load_femnist_datasets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in datasets:
        mask = ds.tensors[1] == target_label
        tensors = [tensor[mask].to(device, non_blocking=True) for tensor in ds.tensors]
        ds.tensors = tensors

    datasets = [ds for ds in datasets if len(ds) > 0]
    train_coleds(cfg, datasets, None)
    experiment_folder = cfg.experiment.folder
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(experiment_folder + "/model_weights.pth"))

    profiler = SingleModelProfiler(model)
    embeddings = []
    for ds in datasets:
        with torch.no_grad():
            embeddings.append(profiler.get_embedding(ds))
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)
    # Compute cosine distances between all pairs of rows
    cosine_distances = cdist(embeddings, embeddings, metric="cosine")

    # Find the row index with the smallest non-diagonal cosine distance to any other row
    np.fill_diagonal(cosine_distances, np.nan)
    min_dist_idx = np.nanargmin(cosine_distances)
    row_idx, _ = np.unravel_index(min_dist_idx, cosine_distances.shape)
    print("Considering row", row_idx)

    # Exclude self-comparison by setting the diagonal to np.nan
    np.fill_diagonal(cosine_distances, np.nan)

    # Find the most similar (minimum distance) and most different (maximum distance) rows
    most_similar_idx = np.nanargmin(cosine_distances[row_idx])
    most_different_idx = np.nanargmax(cosine_distances[row_idx])

    print(f"Row {row_idx} is most similar to row {most_similar_idx} (distance={cosine_distances[row_idx, most_similar_idx]:.4f})")
    print(f"Row {row_idx} is most different from row {most_different_idx} (distance={cosine_distances[row_idx, most_different_idx]:.4f})")

    print(f"Target: {row_idx}")
    print(f"Most similar: {most_similar_idx}")
    print(f"Least similar: {most_different_idx}")

    # Save the selected TensorDatasets in experiment_folder

    target_dataset = datasets[row_idx]
    similar_dataset = datasets[most_similar_idx]
    different_dataset = datasets[most_different_idx]
    torch.save(target_dataset, os.path.join(experiment_folder, "target_dataset.pth"))
    torch.save(similar_dataset, os.path.join(experiment_folder, "similar_dataset.pth"))
    torch.save(different_dataset, os.path.join(experiment_folder, "different_dataset.pth"))


if __name__ == "__main__":
    set_torch_flags()
    load_dotenv()
    run_all()
