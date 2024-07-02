import random

import torch
import torchvision.transforms as T

from flwr_datasets import FederatedDataset


def train_test_split(dataset, test_percentage, seed):
    dataset_len = len(dataset)
    test_size = int(dataset_len * test_percentage)
    train_size = dataset_len - test_size

    return torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )



def partition_dataset(dataset, partitioner, test_percentage, transforms_generator, seed):
    if "femnist" in dataset:
        seed_ = 42
    else:
        seed_ = seed
    fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner}, seed=seed_)

    datasets = []
    idx = 0
    while True:
        try:
            dataset = fds.load_partition(idx)
            if "image" in dataset.column_names:
                dataset = dataset.rename_column("image", "img")
            if "fine_label" in dataset.column_names:
                dataset = dataset.rename_column("fine_label", "label")
            if "character" in dataset.column_names:
                dataset = dataset.rename_column("character", "label")
            dataset = dataset.select_columns(["img", "label"])
        except KeyError:
            break
        transforms = next(transforms_generator)
        dataset = _apply_client_transforms(dataset, transforms)
        dataset.applied_data_transforms = transforms

        tpl = train_test_split(dataset, test_percentage, seed)
        datasets.append(tpl)
        idx += 1
    a, b = 0, 0
    for tr, vl in datasets:
        a += len(tr)
        b += len(vl)
    print(f"Loaded {len(datasets)} datasets")
    print(f"Tot training: {a}")
    print(f"Tot validation: {b}")
    return datasets


def _apply_client_transforms(dataset, transforms):
    return dataset.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")


def _get_random_float(min_=0., max_=1.):
    return random.uniform(min_, max_)


def get_transform_iterator(n_transforms, horizontal_flipping):
    # Define the list of transformations
    transformations = [
        lambda: T.RandomVerticalFlip(p=_get_random_float(0.5, 1.0)),
        lambda: T.ColorJitter(brightness=_get_random_float(), contrast=_get_random_float(), saturation=_get_random_float(), hue=_get_random_float(0., 0.5)),
        lambda: T.RandomRotation(degrees=_get_random_float(0, 90), fill=_get_random_float()),
        lambda: T.ElasticTransform(alpha=_get_random_float(0, 150)),
        lambda: T.GaussianBlur(kernel_size=3, sigma=_get_random_float(0.2, 1.0)),
        lambda: T.RandomInvert(p=_get_random_float()),
        lambda: T.RandomSolarize(_get_random_float(), p=_get_random_float())
    ]

    # random_transform = T.RandomChoice(transformations, )

    if not horizontal_flipping:
        while True:
            yield T.ToTensor()

    while True:
        yield T.Compose(
            [T.ToTensor()] +
            [fn() for fn in random.sample(transformations, n_transforms)] +
            [T.RandomHorizontalFlip()]
        )
