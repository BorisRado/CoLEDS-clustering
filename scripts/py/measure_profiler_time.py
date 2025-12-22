import os
import json
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.utils.other import get_exp_folder, set_torch_flags
from src.profiler.es_profiler import EmbeddingSpaceProfiler
from src.profiler.single_model_profiler import SingleModelProfiler
from src.profiler.wd_profiler import WeightDiffProfiler
from src.profiler.logit_profiler import LogitProfiler



def _get_random_dataset(dataset_size, input_shape):
    # generate random radaset
    return torch.utils.data.TensorDataset(
        torch.randn(dataset_size, *input_shape),
        torch.zeros(dataset_size,).long()
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="execution_time")
def run(cfg):

    print(OmegaConf.to_yaml(cfg))

    INPUT_SHAPE = (3, 32, 32)
    OPTIM_KWARGS = {
        "optimizer_name": "adam",
        "lr": 0.0001,
    }
    set_torch_flags()
    if "folder" in cfg:
        experiment_folder = Path(cfg.folder)
        experiment_folder.mkdir(parents=True, exist_ok=True)
    else:
        experiment_folder = get_exp_folder()

    # initialize the profiler
    profiler_name = cfg.profiler
    if profiler_name == "coleds":
        model = instantiate(cfg.model)
        profiler = SingleModelProfiler(model)

    elif profiler_name == "wdp":
        model = instantiate(cfg.model)
        profiler = WeightDiffProfiler(
            init_model=model,
            ft_epochs=1,
            batch_size=32,
            optim_kwargs=OPTIM_KWARGS,
        )

    elif profiler_name == "es":
        model = instantiate(cfg.model)
        profiler = EmbeddingSpaceProfiler(model=model, statistics=["mean"])

    elif profiler_name == "lgp":
        model = instantiate(cfg.model)
        profiler = LogitProfiler(
            init_model=model,
            ft_epochs=1,
            batch_size=32,
            public_dataset=_get_random_dataset(100, INPUT_SHAPE),
            optim_kwargs=OPTIM_KWARGS,
        )

    else:
        raise Exception(f"Unknown profiler {profiler_name}")

    ds_size = str(cfg.dataset.dataset_size)
    exp_json_filepath = experiment_folder / f"{str(profiler)}_{ds_size}.json"
    assert not os.path.exists(exp_json_filepath)

    print("PROFILER NAME", str(profiler))

    computation_times = []
    for idx in range(cfg.n_evaluations + 1):

        dataset = _get_random_dataset(cfg.dataset.dataset_size, cfg.dataset.input_shape)

        start_time = time.perf_counter()
        with torch.no_grad():
            _ = profiler.get_embedding(dataset)
        end_time = time.perf_counter()

        if idx == 0:
            # do not use the time in the first iteration, may include
            # some initialization overhead that we do not want to measure
            continue

        computation_times.append(end_time - start_time)
        time.sleep(2.0)  # to cool down a bit
        print(idx)

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
