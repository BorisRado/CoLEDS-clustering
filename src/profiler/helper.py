from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.profiler.single_model_profiler import SingleModelProfiler
from src.profiler.es_profiler import EmbeddingSpaceProfiler
from src.profiler.wd_profiler import WeightDiffProfiler
from src.profiler.logit_profiler import LogitProfiler
from src.profiler.label_dist_profiler import LabelProfiler
from src.data.utils import get_holdout_dataset



def load_coleds_profiler(profiler_config, folder):
    coleds_model = instantiate(profiler_config.model)
    coleds_model.load_state_dict(
        torch.load(folder / "model_weights.pth"), strict=True
    )
    return SingleModelProfiler(coleds_model)


def load_es_profiler(profiler_config, folder):
    encoder_model = instantiate(profiler_config.model)
    encoder_model.load_state_dict(
        torch.load(folder / "fl_model.pth"), strict=True
    )
    return EmbeddingSpaceProfiler(
        model=encoder_model,
        statistics=OmegaConf.to_container(profiler_config.profiling.statistics)
    )


def load_wd_profiler(profiler_config, folder):
    encoder_model = instantiate(profiler_config.model)
    fl_model_path = folder / "fl_model.pth"
    if fl_model_path.exists():
        print("Loading pre-existing weights")
        encoder_model.load_state_dict(torch.load(fl_model_path), strict=True)
    return WeightDiffProfiler(
        init_model=encoder_model,
        **OmegaConf.to_container(profiler_config.profiling),
        optim_kwargs=OmegaConf.to_container(profiler_config.optimizer),
    )


def load_logit_profiler(profiler_config, folder):
    encoder_model = instantiate(profiler_config.model)
    fl_model_path = folder / "fl_model.pth"
    public_dataset = get_holdout_dataset(profiler_config.public_dataset)
    if fl_model_path.exists():
        print("Loading pre-existing weights")
        encoder_model.load_state_dict(torch.load(fl_model_path), strict=True)
    return LogitProfiler(
        init_model=encoder_model,
        public_dataset=public_dataset,
        **OmegaConf.to_container(profiler_config.profiling),
        optim_kwargs=OmegaConf.to_container(profiler_config.optimizer),
    )


def load_profiler(cfg):
    folder = Path(cfg.folder)
    profiler_config = OmegaConf.load(folder / ".hydra" / "config.yaml")
    output_folder = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if "train_config" in cfg:
        assert cfg.dataset.dataset_name == profiler_config.dataset.dataset_name

    profiler_type = profiler_config["profiler"]
    if profiler_type == "coleds":
        profiler = load_coleds_profiler(profiler_config, folder)

    elif profiler_type == "es":
        profiler = load_es_profiler(profiler_config, folder)

    elif profiler_type == "wd":
        profiler = load_wd_profiler(profiler_config, folder)

    elif profiler_type == "logit":
        profiler = load_logit_profiler(profiler_config, folder)

    elif profiler_type == "label":
        profiler = LabelProfiler(cfg.dataset.n_classes)

    return profiler, output_folder
