import os
import json
import time
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import OmegaConf
from tqdm import tqdm

from src.cem.helper import load_cem
from src.utils.other import get_exp_folder, set_torch_flags


def _get_random_dataset(cfg):
    # generate random radaset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(cfg.dataset.dataset_size+1, *cfg.dataset.input_shape),
        torch.zeros(cfg.dataset.dataset_size+1,).long()
    )
    dataset = torch.utils.data.random_split(
        dataset,
        [cfg.dataset.dataset_size, 1]
    )[0]
    return dataset


@hydra.main(version_base=None, config_path="../conf", config_name="execution_time")
def run(cfg):

    print(OmegaConf.to_yaml(cfg))

    set_torch_flags()
    if "folder" in cfg:
        experiment_folder = Path(cfg.folder)
        experiment_folder.mkdir(parents=True, exist_ok=True)
    else:
        experiment_folder = get_exp_folder()

    cem = load_cem(cfg, cem_folder=None)
    ds_size = str(cfg.dataset.dataset_size)
    exp_json_filepath = experiment_folder / f"{str(cem)}_{ds_size}.json"
    assert not os.path.exists(exp_json_filepath)

    print("CEM NAME", str(cem))

    computation_times = []
    for idx in tqdm(range(cfg.n_evaluations + 1)):

        dataset = _get_random_dataset(cfg)

        start_time = time.time()
        with torch.no_grad():
            _ = cem.get_embedding(dataset)
        end_time = time.time()
        if idx == 0:
            # do not use the time in the first iteration, may include
            # some initialization overhead that we do not want to measure
            continue
        computation_times.append(end_time - start_time)

        time.sleep(2)  # to cool down a bit

    data = {
        "GPU": torch.cuda.is_available(),
        "times": computation_times,
        "config": OmegaConf.to_container(cfg),
    }
    with open(exp_json_filepath, "w") as fp:
        json.dump(data, fp, indent=4)

    print(computation_times)


if __name__ == "__main__":
    run()
