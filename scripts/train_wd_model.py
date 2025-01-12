from functools import partial

import hydra
from hydra.utils import instantiate

from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg, to_pytorch_tensor_dataset
from src.utils.evaluation import eval_fn
from src.utils.stochasticity import set_seed, TempRng
from src.utils.wandb import init_wandb, finish_wandb
from src.models.training_procedures import train_ce
from src.flower.train import train_flower
from src.utils.other import get_exp_folder, set_torch_flags


@hydra.main(version_base=None, config_path="../conf", config_name="wd")
def run(cfg):
    set_torch_flags()
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)
    experiment_folder = get_exp_folder()

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainsets = to_pytorch_tensor_dataset(trainsets)
    valsets = to_pytorch_tensor_dataset(valsets)

    with TempRng(cfg.general.seed):
        model = instantiate(
            cfg.model,
            input_shape=cfg.dataset.input_shape,
            n_classes=cfg.dataset.n_classes
        )

    if cfg.dataset.dataset_name != "synthetic":
        eval_fn_ = partial(
            eval_fn,
            trainsets=trainsets,
            valsets=valsets,
            n_classes=cfg.dataset.n_classes,
            experiment_folder=experiment_folder,
        )
    else:
        eval_fn_ = lambda *args, **kwargs: 1


    cem_fn = instantiate(
        cfg.cem,
        optim_kwargs=OmegaConf.to_container(cfg.optimizer),
        _partial_=True
    )
    eval_fn_(
        cem=cem_fn(init_model=model),
        iter=-1
    )

    for idx in range(2):

        model = train_flower(
            model,
            client_fn_kwargs={
                "trainsets": trainsets,
                "valsets": valsets,
                "batch_size": 32,
                "train_fn": train_ce,
            },
            optim_kwargs=OmegaConf.to_container(cfg.optimizer),
            n_rounds=10,
            experiment_folder=experiment_folder,
            strategy_kwargs={},
            seed=cfg.general.seed,
        )
        eval_fn_(
            cem=cem_fn(init_model=model),
            iter=idx
        )
    finish_wandb()


if __name__ == "__main__":
    run()
