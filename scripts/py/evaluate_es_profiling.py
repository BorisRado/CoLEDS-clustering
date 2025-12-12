from functools import partial
from dotenv import load_dotenv

import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf

from src.data.utils import get_datasets_from_cfg, to_pytorch_tensor_dataset
from src.utils.evaluation import get_evaluation_fn
from src.utils.stochasticity import set_seed
from src.utils.wandb import init_wandb, finish_wandb, run_exists_already
from src.models.training_procedures import train_ce, train_vae
from src.flower.train import train_flower
from src.utils.other import get_exp_folder, set_torch_flags
from src.profiler.es_profiler import EmbeddingSpaceProfiler


@hydra.main(version_base=None, config_path="../../conf", config_name="es")
def run(cfg):
    OmegaConf.resolve(cfg)
    if run_exists_already(cfg):
        print("Run exists already. Returning...")
        return
    experiment_folder = get_exp_folder()
    cfg.experiment.folder = str(experiment_folder)
    print(OmegaConf.to_yaml(cfg))
    init_wandb(cfg)

    set_seed(cfg.general.seed)
    trainsets, valsets = get_datasets_from_cfg(cfg)
    trainsets = to_pytorch_tensor_dataset(trainsets, n_classes=cfg.dataset.n_classes)
    valsets = to_pytorch_tensor_dataset(valsets, n_classes=cfg.dataset.n_classes)

    model = instantiate(cfg.model)
    assert "encoder" in model
    if "recon_head" in model:
        print("Training autoencoder")
        train_fn = train_vae
    elif "clf_head" in model:
        print("Training classification head")
        train_fn = train_ce
    else:
        raise Exception("Invalid model")

    eval_fn = get_evaluation_fn(cfg, trainsets, valsets, experiment_folder)

    profiler_fn = partial(
        EmbeddingSpaceProfiler,
        statistics=OmegaConf.to_container(cfg.profiling.statistics),
    )

    best_correlation = eval_fn(
        profiler=profiler_fn(model=model),
        iter=-1
    )

    for idx in range(cfg.general.eval_iterations):
        print(f"Best correlation: {best_correlation}")

        model = train_flower(
            model,
            client_fn_kwargs={
                "trainsets": trainsets,
                "valsets": valsets,
                "batch_size": cfg.train_config.batch_size,
                "train_fn": train_fn,
            },
            optim_kwargs=OmegaConf.to_container(cfg.optimizer),
            n_rounds=cfg.general.epochs_per_iteration,
            experiment_folder=experiment_folder,
            strategy_kwargs={},
            seed=cfg.general.seed,
        )
        tmp_corr = eval_fn(
            profiler=profiler_fn(model=model),
            iter=idx
        )
        print("Got correlation: ", tmp_corr)
        if tmp_corr > best_correlation:
            best_correlation = tmp_corr

    print(f"Best correlation: {best_correlation}")
    finish_wandb()


if __name__ == "__main__":
    set_torch_flags()
    load_dotenv()
    run()
