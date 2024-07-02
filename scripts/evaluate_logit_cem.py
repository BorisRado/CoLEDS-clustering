from functools import partial

import hydra
from hydra.utils import instantiate

from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg
from src.utils.evaluation import eval_fn
from src.utils.stochasticity import set_seed, TempRng
from src.utils.wandb import init_wandb, finish_wandb
from src.models.training_procedures import train_ce
from src.flower.train import train_flower



@hydra.main(version_base=None, config_path="../conf", config_name="logit")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    experiment_folder = init_wandb(cfg)

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)

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

    finish_wandb()


if __name__ == "__main__":
    run()
