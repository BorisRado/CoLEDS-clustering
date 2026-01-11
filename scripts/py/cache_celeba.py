import os
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import torch
import torchvision.transforms as T
from tqdm import tqdm
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import TensorDataset
from datasets.utils.logging import disable_progress_bar

NUM_DATASETS = 5_000
HOME_FOLDER = "data/raw/celeba"

COLUMNS = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young"
]


def _apply_client_transforms(dataset, transforms):
    return dataset.map(
        lambda img: {"image": transforms(img)}, input_columns="image"
    ).with_format("torch")


def run():

    start_time = time.time()

    os.makedirs(HOME_FOLDER, exist_ok=True)
    partitioner = NaturalIdPartitioner(partition_by="celeb_id")
    fds = FederatedDataset(
        dataset="flwrlabs/celeba",
        partitioners={"train": partitioner},
        seed=1602 # determines the order of data inside a dataset, not the order of the datasets
    )

    transforms = T.Compose([
        T.CenterCrop(178),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for idx in tqdm(range(NUM_DATASETS), total=NUM_DATASETS):
        dataset = fds.load_partition(idx)
        dataset = _apply_client_transforms(dataset, transforms)

        imgs = []
        labels = defaultdict(list)
        for ex in dataset:
            imgs.append(ex["image"])
            for col in COLUMNS:
                lbl = ex[col]
                labels[col].append(lbl.item() if isinstance(lbl, torch.Tensor) else int(lbl))

        imgs = torch.stack(imgs)
        labels = (torch.tensor(l) for l in labels.values())
        dataset = TensorDataset(imgs, *labels)
        torch.save(dataset, f"{HOME_FOLDER}/{idx}.pth")
    print(f"Caching time: {time.time() - start_time}")


def load_datasets():
    start_time = time.time()
    datasets = []
    for idx in range(NUM_DATASETS):
        ds = torch.load(f"{HOME_FOLDER}/{idx}.pth", weights_only=True)
        datasets.append(ds)
    print(f"Loading time: {time.time() - start_time}")
    return datasets


if __name__ == "__main__":
    # Allowlist TensorDataset for safe weights-only unpickling
    # This permits `torch.load(..., weights_only=True)` to succeed for files
    # containing a `TensorDataset` without requiring `weights_only=False`.
    torch.serialization.add_safe_globals([TensorDataset])
    disable_progress_bar()

    run()

    # laod the datasets to compare the speed
    load_datasets()
