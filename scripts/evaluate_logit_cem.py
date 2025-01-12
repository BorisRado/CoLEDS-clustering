import hydra
from hydra.utils import instantiate

from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg, to_pytorch_tensor_dataset
from src.flower.train import train_flower
from src.utils.evaluation import get_evaluation_fn
from src.utils.stochasticity import TempRng
from src.utils.wandb import init_wandb, finish_wandb
from src.models.training_procedures import train_ce
from src.utils.other import get_exp_folder, set_torch_flags


@hydra.main(version_base=None, config_path="../conf", config_name="logit")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)
    experiment_folder = get_exp_folder()
    set_torch_flags()

    with TempRng(cfg.general.seed):
        trainsets, valsets = get_datasets_from_cfg(cfg)

    trainsets = to_pytorch_tensor_dataset(trainsets)
    valsets = to_pytorch_tensor_dataset(valsets)

    with TempRng(cfg.general.seed):
        model = instantiate(
            cfg.model,
            input_shape=cfg.dataset.input_shape,
            n_classes=cfg.dataset.n_classes
        )

    eval_fn = get_evaluation_fn(cfg, trainsets, valsets, experiment_folder)

    cem_fn = instantiate(
        cfg.cem,
        optim_kwargs=OmegaConf.to_container(cfg.optimizer),
        _partial_=True
    )
    eval_fn(cem=cem_fn(init_model=model), iter=-1)

    train_flower(
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
    eval_fn(cem=cem_fn(init_model=model), iter=0)
    finish_wandb()


if __name__ == "__main__":
    run()
