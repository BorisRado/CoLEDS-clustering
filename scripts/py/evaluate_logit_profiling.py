from functools import partial
from dotenv import load_dotenv

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf
import torch

from src.data.utils import get_datasets_from_cfg, to_pytorch_tensor_dataset, get_holdout_dataset
from src.flower.train import train_flower
from src.utils.evaluation import get_evaluation_fn
from src.utils.stochasticity import set_seed
from src.utils.wandb import init_wandb, finish_wandb
from src.models.training_procedures import train_ce
from src.utils.other import get_exp_folder, set_torch_flags
from src.profiler.logit_profiler import LogitProfiler


@hydra.main(version_base=None, config_path="../../conf", config_name="logit")
def run(cfg):
    OmegaConf.resolve(cfg)
    experiment_folder = get_exp_folder()
    cfg.experiment.folder = str(experiment_folder)
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)

    set_seed(cfg.general.seed)

    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainsets = to_pytorch_tensor_dataset(trainsets, n_classes=cfg.dataset.n_classes)
    valsets = to_pytorch_tensor_dataset(valsets, n_classes=cfg.dataset.n_classes)

    model = instantiate(cfg.model)

    ho_dataset = get_holdout_dataset(cfg.public_dataset)
    idxs = torch.randperm(len(ho_dataset))[:cfg.profiling.public_dataset_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ho_dataset.tensors = tuple(tensor[idxs].to(device) for tensor in ho_dataset.tensors)

    eval_fn = get_evaluation_fn(cfg, trainsets, valsets, experiment_folder)

    profiling_cfg = OmegaConf.to_container(cfg.profiling)
    profiling_cfg.pop("public_dataset_size")
    profiler_fn = partial(LogitProfiler,
        public_dataset=ho_dataset,
        optim_kwargs=OmegaConf.to_container(cfg.optimizer),
        **profiling_cfg
    )
    best_correlation = eval_fn(profiler=profiler_fn(init_model=model), iter=-1)
    print(f"Correlation: {best_correlation}")

    for _ in range(cfg.general.eval_iterations):
        model = train_flower(
            model,
            client_fn_kwargs={
                "trainsets": trainsets,
                "valsets": valsets,
                "batch_size": cfg.train_config.batch_size,
                "train_fn": train_ce
            },
            optim_kwargs=OmegaConf.to_container(cfg.optimizer),
            n_rounds=cfg.train_config.train_epochs,
            experiment_folder=experiment_folder,
            strategy_kwargs={"fraction_fit": 0.5, "fraction_evaluate": 1.0},
            seed=cfg.general.seed
        )
        tmp_corr = eval_fn(profiler=profiler_fn(init_model=model), iter=0)
        print("Got correlation: ", tmp_corr)
        if tmp_corr > best_correlation:
            best_correlation = tmp_corr

    print(f"Best correlation: {best_correlation}")
    finish_wandb()


if __name__ == "__main__":
    load_dotenv()
    set_torch_flags()
    run()
