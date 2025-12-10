from hydra.utils import instantiate
import torch
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar


disable_progress_bar()


def train_test_split(dataset, test_percentage, seed):
    dataset_len = len(dataset)
    test_size = int(dataset_len * test_percentage)
    train_size = dataset_len - test_size

    return torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )



def partition_dataset(dataset, partitioner, test_percentage, transforms, seed):
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
        except (KeyError,ValueError):
            break
        dataset = _apply_client_transforms(dataset, transforms)

        # tpl = train_test_split(dataset, test_percentage, seed)
        tpl = dataset.train_test_split(test_size=test_percentage, seed=seed)
        trainset = tpl["train"]
        testset = tpl["test"]
        trainset.applied_data_transforms = transforms
        testset.applied_data_transforms = transforms
        datasets.append((trainset, testset))
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
