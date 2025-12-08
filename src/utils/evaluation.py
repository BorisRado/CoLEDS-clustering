from functools import partial

import wandb

from src.utils.stochasticity import TempRng
from src.utils.wandb import log_table
from src.testing.test_all import test_all


def eval_fn(cem, trainsets, valsets, experiment_folder, iter, **kwargs):
    with TempRng(1602):
        results = test_all(cem, trainsets, valsets, **kwargs)
    cem_name = cem.__class__.__name__
    for k, values in results.items():
        log_table(values, experiment_folder, f"{cem_name}_{k}_{iter}", iter=iter)
    corr = results["correlation"][0]["correlation"]
    if wandb.run is not None:
        wandb.log({"correlation": corr, "iter": iter})
    return corr


def get_evaluation_fn(cfg, trainsets, valsets, experiment_folder):
    if cfg.dataset.dataset_name != "synthetic":
        eval_fn_ = partial(
            eval_fn,
            trainsets=trainsets,
            valsets=valsets,
            experiment_folder=experiment_folder,
        )
    else:
        eval_fn_ = lambda *args, **kwargs: 1
    return eval_fn_
