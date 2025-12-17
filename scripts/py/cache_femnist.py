import time
from functools import partial
from multiprocessing import Pool

import torch
import torchvision.transforms as T
from tqdm import tqdm
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import TensorDataset
from datasets.utils.logging import disable_progress_bar

NUM_DATASETS = 3597
HOME_FOLDER = "data/raw/femnist"


def _apply_client_transforms(dataset, transforms):
    return dataset.map(
        lambda img: {"image": transforms(img)}, input_columns="image"
    ).with_format("torch")


def run():

    start_time = time.time()

    partitioner = NaturalIdPartitioner(partition_by="writer_id")
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": partitioner},
        seed=1602 # determines the order of data inside a dataset, not the order of the datasets
    )

    transforms = T.ToTensor()

    for idx in tqdm(range(NUM_DATASETS), total=NUM_DATASETS):
        dataset = fds.load_partition(idx)
        dataset = _apply_client_transforms(dataset, transforms)

        imgs = []
        labels = []
        for ex in dataset:
            imgs.append(ex["image"])
            lbl = ex["character"]
            labels.append(lbl.item() if isinstance(lbl, torch.Tensor) else int(lbl))

        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        dataset = TensorDataset(imgs, labels)
        torch.save(dataset, f"{HOME_FOLDER}/{idx}.pth")
    print(f"Caching time: {time.time() - start_time}")


def load_datasets():
    start_time = time.time()
    for idx in range(NUM_DATASETS):
        _ = torch.load(f"{HOME_FOLDER}/{idx}.pth", weights_only=True)
    print(f"Loading time: {time.time() - start_time}")


if __name__ == "__main__":
    # Allowlist TensorDataset for safe weights-only unpickling
    # This permits `torch.load(..., weights_only=True)` to succeed for files
    # containing a `TensorDataset` without requiring `weights_only=False`.
    torch.serialization.add_safe_globals([TensorDataset])
    disable_progress_bar()

    run()

    # laod the datasets to compare the speed
    load_datasets()
