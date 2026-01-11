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
from scripts.py.cache_celeba import load_datasets, COLUMNS
from src.profiler.single_model_profiler import SingleModelProfiler
from scipy.spatial.distance import cdist


@hydra.main(version_base=None, config_path="../../conf", config_name="coleds")
def run_all(cfg):

    cfg.experiment.save_model = True
    torch.serialization.add_safe_globals([torch.utils.data.TensorDataset])
    datasets = load_datasets()

    reduced_datasets = []
    for ds in datasets:
        # create a new tensors dataset with only one label, otherwise error in training logic
        new_dataset = torch.utils.data.TensorDataset(
            ds.tensors[0].cuda(), ds.tensors[1].cuda()
        )
        reduced_datasets.append(new_dataset)

    train_coleds(cfg, reduced_datasets, None)
    del reduced_datasets
    for ds in datasets:
        ds.tensors = [t.cuda() for t in ds.tensors]

    experiment_folder = cfg.experiment.folder
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(experiment_folder + "/model_weights.pth"))

    # compute the final embeddings
    profiler = SingleModelProfiler(model)
    embeddings = []
    for ds in datasets:
        with torch.no_grad():
            embeddings.append(profiler.get_embedding(ds))
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)

    # Compute cosine distances between all pairs of rows
    pred_cosine_distances = cdist(embeddings, embeddings, metric="cosine")
    sim1 = pred_cosine_distances[np.triu_indices(pred_cosine_distances.shape[0], 1)]

    # compute the correlation with every column
    for idx, column in enumerate(COLUMNS, 1):
        num_positives = [(ds.tensors[idx].sum(), len(ds)) for ds in datasets]
        label_distributions = [torch.tensor([n_pos, size - n_pos], dtype=torch.float32) for n_pos, size in num_positives]
        label_cosine_distances = cdist(label_distributions, label_distributions, metric="cosine")

        sim2 = label_cosine_distances[np.triu_indices(label_cosine_distances.shape[0], 1)]

        assert sim1.ndim == 1 and sim1.shape == sim2.shape
        correlation = np.corrcoef(sim1, sim2)[0, 1]
        print(column, correlation)

    # Find the row index with the smallest non-diagonal cosine distance to any other row
    np.fill_diagonal(pred_cosine_distances, np.nan)
    for _ in range(15):
        row_idx = torch.randint(len(datasets), (1,)).item()
        print("Considering row", row_idx)

        # Exclude self-comparison by setting the diagonal to np.nan
        np.fill_diagonal(pred_cosine_distances, np.nan)

        # Define N for top-N similar/different datasets
        N = 5  # Adjust as needed

        # Get sorted indices for most similar (smallest distances) and most different (largest distances)
        distances_from_row = pred_cosine_distances[row_idx]
        valid_indices = ~np.isnan(distances_from_row)
        valid_distances = distances_from_row[valid_indices]
        valid_idx_mapping = np.where(valid_indices)[0]

        # Get N most similar (smallest distances)
        most_similar_sorted = np.argsort(valid_distances)[:N]
        most_similar_indices = valid_idx_mapping[most_similar_sorted]

        # Get N most different (largest distances)
        most_different_sorted = np.argsort(valid_distances)[-N:][::-1]
        most_different_indices = valid_idx_mapping[most_different_sorted]

        # Get N closest to average distance (0 for cosine distance)
        abs_distances = np.abs(0.5 - valid_distances)
        closest_to_avg_sorted = np.argsort(abs_distances)[:N]
        closest_to_avg_indices = valid_idx_mapping[closest_to_avg_sorted]

        print(f"\nTarget row: {row_idx}")
        print(f"\nTop {N} most similar rows:")
        for i, idx in enumerate(most_similar_indices, 1):
            print(f"  {i}. Row {idx} (distance={pred_cosine_distances[row_idx, idx]:.4f})")

        print(f"\nTop {N} most different rows:")
        for i, idx in enumerate(most_different_indices, 1):
            print(f"  {i}. Row {idx} (distance={pred_cosine_distances[row_idx, idx]:.4f})")

        print(f"\nTop {N} closest to average distance:")
        for i, idx in enumerate(closest_to_avg_indices, 1):
            print(f"  {i}. Row {idx} (distance={pred_cosine_distances[row_idx, idx]:.4f})")

        # Save the target dataset
        target_dataset = datasets[row_idx]
        torch.save(target_dataset, os.path.join(experiment_folder, f"target_dataset_{row_idx}.pth"))

        # Save N most similar datasets
        similar_datasets = [datasets[idx] for idx in most_similar_indices]
        torch.save(similar_datasets, os.path.join(experiment_folder, f"top_similar_datasets_{row_idx}.pth"))

        # Save N most different datasets
        different_datasets = [datasets[idx] for idx in most_different_indices]
        torch.save(different_datasets, os.path.join(experiment_folder, f"top_different_datasets_{row_idx}.pth"))

        # Save N closest to average distance datasets
        avg_distance_datasets = [datasets[idx] for idx in closest_to_avg_indices]
        torch.save(avg_distance_datasets, os.path.join(experiment_folder, f"top_avg_distance_datasets_{row_idx}.pth"))


if __name__ == "__main__":
    set_torch_flags()
    load_dotenv()
    run_all()
