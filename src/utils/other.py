from pathlib import Path
from typing import List, Iterator
import itertools

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig


def get_exp_folder():
    exp_folder = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print("Experiment folder: ", exp_folder)
    return exp_folder


def set_torch_flags():
    # these flags help us to achieve better numerical stability across experiments and hence
    # help achieving more reproducible results
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_flush_denormal(True)


def _set_nested_value(cfg: DictConfig, key: str, value):
    keys = key.split(".")
    current = cfg
    for k in keys[:-1]:
        if k not in current:
            current[k] = OmegaConf.create({})
        current = current[k]
    current[keys[-1]] = value


def iterate_configs(cfg: DictConfig, multirun_columns: List[str]) -> Iterator[DictConfig]:
    """
    Generates all possible combinations of configurations based on multirun columns.

    For each column in multirun_columns that contains a list, generates all combinations
    of those list values while keeping non-list values unchanged.
    """
    # Extract values for multirun columns
    multirun_values = []
    multirun_keys = []

    for column in multirun_columns:
        value = OmegaConf.select(cfg, column)
        multirun_keys.append(column)
        if isinstance(value, ListConfig):
            multirun_values.append(OmegaConf.to_container(value))
        elif isinstance(value, (list, tuple)):
            multirun_values.append(value)
        else:
            assert isinstance(value, (int, float, str, bool)), str(type(value))
            # If it's not a list, treat it as a single-item list
            multirun_values.append([value])

    # If no multirun columns found, yield the original config
    if not multirun_values:
        yield cfg
        return

    # assert all([len(v) == len(set(v)) for v in multirun_values])
    # Generate all combinations using itertools.product
    for combination in itertools.product(*multirun_values):
        # Create a deep copy of the original config
        new_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))

        # Update the multirun columns with the current combination
        for key, value in zip(multirun_keys, combination):
            # Handle hierarchical keys like 'a.b.c' by setting them directly on the config
            _set_nested_value(new_cfg, key, value)

        yield new_cfg
