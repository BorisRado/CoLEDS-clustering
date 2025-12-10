import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as DS
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset
from hydra.utils import instantiate

from src.utils.stochasticity import TempRng
from src.data.synthetic_dataset import generate_dataset
from src.data.partitioning import partition_dataset, train_test_split
from src.data.synthetic_dataset import generate_synthetic_datasets
import torch.multiprocessing as mp


def get_dataloaders_with_replacement(datasets, batch_size, runtime_horizontal_flipping):

    def collate_fn(batch):
        # batch = [(img, label), (img, label), ...]
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels, device=imgs.device)
        mask = torch.rand(len(imgs), device=imgs.device) < 0.5
        imgs[mask] = torch.flip(imgs[mask], dims=[-1])
        return imgs, labels

    dataloaders = []
    for dataset in datasets:
        sampler = RandomSampler(dataset, replacement=True, num_samples=2*batch_size)
        kwargs = {"collate_fn": collate_fn} if runtime_horizontal_flipping else {}
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, **kwargs)
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
        v[batch[1]] += 1
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
        datasets = []
        for idx in range(3597):
            ds = torch.load(f"/home/radovib/femnist/{idx}.pth")
            ds.set_format("torch")
            ds.info.dataset_name = "femnist"
            datasets.append(ds)
        datasets = [
            train_test_split(ds, cfg.dataset.test_percentage, cfg.general.seed)
            for ds in datasets
        ]
        trainsets, valsets = zip(*datasets)
        return list(trainsets), list(valsets)

    partitioner = instantiate(cfg.partitioning)

    transforms = instantiate(cfg.dataset.transforms)
    datasets = partition_dataset(
        cfg.dataset.dataset_name,
        partitioner,
        cfg.dataset.test_percentage,
        transforms=transforms,
        seed=cfg.general.seed
    )

    trainsets, valsets = zip(*datasets)
    return trainsets, valsets


def get_holdout_dataset(config):
    dataset_name = config.dataset_name
    if config.dataset_name == "synthetic":
        with TempRng(58):
            dataset = generate_dataset(50, -1.0, dictionary=False)
        return dataset

    holdout_kwargs = {
        "root": Path(torch.hub.get_dir()) / "datasets",
        "train": False,
        "download": False,
        "transform": instantiate(config.transforms)
    }

    holdout_dataset = {
        "cifar10": DS.CIFAR10,
        "cifar100": DS.CIFAR100,
        "mnist": DS.MNIST,
        "fashion_mnist": DS.FashionMNIST
    }[dataset_name](**holdout_kwargs)

    # load it into memory
    imgs, labels = [], []
    dl = DataLoader(holdout_dataset, batch_size=16)
    for b in dl:
        imgs.append(b[0])
        labels.append(b[1])
    imgs = torch.vstack(imgs)
    labels = torch.hstack(labels)
    holdout_dataset = TensorDataset(imgs, labels)
    return holdout_dataset


def get_dataset_target_indices(dataset):
    out = {}
    for idx, (_, target) in enumerate(dataset):
        if not isinstance(target, int):
            target = target.item()
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


def _process_dataset(ds, n_classes):

    dl = DataLoader(
        ds, batch_size=256, drop_last=False,
        pin_memory=torch.cuda.is_available(), shuffle=False,
    )

    # Determine device and dataset size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_size = len(ds)

    # Get first batch to determine tensor shapes
    img_shape = next(iter(dl))["img"].shape[1:]  # Remove batch dimension

    # Pre-allocate tensors on GPU if available
    imgs = torch.empty((dataset_size, *img_shape), device=device)
    labels = torch.empty(dataset_size, dtype=torch.long, device=device)

    # Fill tensors with data
    idx = 0
    label_distribution = np.zeros((n_classes,))
    for b in dl:
        batch_size = b["img"].shape[0]
        imgs[idx:idx+batch_size] = b["img"].to(device, non_blocking=True)
        labels[idx:idx+batch_size] = b["label"].to(device, non_blocking=True)
        for l in labels[idx:idx+batch_size]:
            label_distribution[l] += 1

        idx += batch_size

    tensor_ds = TensorDataset(imgs, labels)
    tensor_ds.applied_data_transform = ds.applied_data_transforms
    tensor_ds.dataset_name = ds.info.dataset_name
    tensor_ds._label_distribution = label_distribution
    return tensor_ds

def to_pytorch_tensor_dataset(datasets, n_classes):
    # cuda_datasets_seq = [_process_dataset(ds, n_classes) for ds in datasets]

    # Process on CPU in parallel
    start_time = time.time()
    with mp.Pool(processes=8) as pool:
        results = pool.starmap(_process_dataset_cpu, [(ds, n_classes) for ds in datasets])

    # Move to GPU in main process
    cuda_datasets = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for imgs, labels, label_dist, transforms, dataset_name in results:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        tensor_ds = TensorDataset(imgs, labels)
        tensor_ds.applied_data_transform = transforms
        tensor_ds.dataset_name = dataset_name
        tensor_ds._label_distribution = label_dist
        cuda_datasets.append(tensor_ds)

    print(f"Loading time on GPU: {time.time() - start_time:.2f}")
    # for ds1, ds2 in zip(cuda_datasets, cuda_datasets_seq):
    #     assert len(ds1) == len(ds2)
    #     for dp1, dp2 in zip(ds1, ds2):
    #         assert (dp1[0] == dp2[0]).all()
    #         assert (dp1[1] == dp2[1]).all()
    return cuda_datasets


class ToRgb(torch.nn.Module):
    def forward(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


def _process_dataset_cpu(ds, n_classes):
    """Process dataset on CPU, return raw tensors"""
    dl = DataLoader(
        ds, batch_size=min(128, len(ds)), drop_last=False,
        pin_memory=False, shuffle=False,  # Don't pin memory in subprocess
    )

    dataset_size = len(ds)
    img_shape = next(iter(dl))["img"].shape[1:]

    imgs = torch.empty((dataset_size, *img_shape))
    labels = torch.empty(dataset_size, dtype=torch.long)

    idx = 0
    label_distribution = np.zeros((n_classes,))
    for b in dl:
        batch_size = b["img"].shape[0]
        imgs[idx:idx+batch_size] = b["img"]
        labels[idx:idx+batch_size] = b["label"]
        for l in labels[idx:idx+batch_size]:
            label_distribution[l] += 1
        idx += batch_size

    return imgs, labels, label_distribution, ds.applied_data_transforms, ds.info.dataset_name
