from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as DS
from torch.utils.data import DataLoader, RandomSampler, Subset
from hydra.utils import instantiate

from src.utils.stochasticity import TempRng
from src.data.synthetic_dataset import generate_dataset
from src.data.partitioning import partition_dataset, get_transform_iterator, train_test_split
from src.data.synthetic_dataset import generate_synthetic_datasets


def get_dataloaders_with_replacement(datasets, batch_size):
    dataloaders = []
    for dataset in datasets:
        if len(dataset) > batch_size * 1.5:
            bs = batch_size
        else:
            bs = int(len(dataset) * 2 / 3)
        sampler = RandomSampler(dataset, replacement=True, num_samples=2*bs)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=bs)
        dataloaders.append(dataloader)
    return dataloaders


def get_dataloaders(datasets, batch_size=None, **kwargs):
    dataloaders = []
    for dataset in datasets:
        bs = batch_size if batch_size is not None else len(dataset)
        dataloader = DataLoader(dataset, batch_size=bs, **kwargs)
        dataloaders.append(dataloader)
    return dataloaders


def get_label_distribution(dataset, n_classes):
    v = np.zeros((n_classes,))

    for batch in dataset:
        v[batch["label"]] += 1
    return v


def get_datasets_from_cfg(cfg):
    if cfg.dataset.dataset_name == "synthetic":
        datasets = generate_synthetic_datasets(
            cfg.dataset.n_datasets,
            cfg.dataset.dataset_size,
            p=cfg.dataset.p
        )
        datasets = [
            train_test_split(ds, cfg.dataset.test_percentage, cfg.general.seed)
            for ds in datasets
        ]
        trainsets, valsets = zip(*datasets)
        return list(trainsets), list(valsets)

    if cfg.dataset.dataset_name == "femnist":
        print("Anton")
        datasets = []
        for idx in range(3597):
            ds = torch.load(f"/home/radovib/femnist/{idx}.pth")
            ds.set_format("torch")
            datasets.append(ds)
        datasets = [
            train_test_split(ds, cfg.dataset.test_percentage, cfg.general.seed)
            for ds in datasets
        ]
        trainsets, valsets = zip(*datasets)
        return list(trainsets), list(valsets)

    partitioner = instantiate(cfg.partitioning)

    transforms = get_transform_iterator(0, cfg.dataset.dataset_name != "mnist")
    datasets = partition_dataset(
        cfg.dataset.dataset_name,
        partitioner,
        cfg.dataset.test_percentage,
        transforms_generator=transforms,
        seed=cfg.general.seed
    )

    trainsets, valsets = zip(*datasets)
    return trainsets, valsets


def get_holdout_dataset(dataset_name, to_tensor=False):
    if dataset_name == "synthetic":
        with TempRng(58):
            dataset = generate_dataset(50, -1.0, dictionary=False)
        return dataset

    holdout_kwargs = {
        "root": Path(torch.hub.get_dir()) / "datasets",
        "train": False,
        "download": False,
    }
    if to_tensor:
        holdout_kwargs["transform"] = T.ToTensor()

    holdout_dataset = {
        "cifar10": DS.CIFAR10,
        "cifar100": DS.CIFAR100,
        "mnist": DS.MNIST,
        "fashion_mnist": DS.FashionMNIST
    }[dataset_name](**holdout_kwargs)
    return holdout_dataset


def get_dataset_target_indices(dataset):
    out = {}
    for idx, (_, target) in enumerate(dataset):
        if target not in out:
            out[target] = []
        out[target].append(idx)
    return out


def sample_subset_given_distribution(dataset, indices, target_distribution):
    idxs = []
    for idx, num in enumerate(target_distribution):
        idxs.extend(np.random.choice(indices[idx], size=num, replace=False))
    subset = Subset(dataset, idxs)
    return subset
