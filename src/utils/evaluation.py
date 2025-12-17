from functools import partial

import wandb

from src.utils.stochasticity import TempRng
from src.utils.wandb import log_table
from src.testing.test_all import test_all


def eval_fn(profiler, trainsets, valsets, experiment_folder, iter, **kwargs):
    with TempRng(1602):
        results = test_all(profiler, trainsets, valsets, **kwargs)
    profiler_name = profiler.__class__.__name__
    for k, values in results.items():
        log_table(values, experiment_folder, f"{profiler_name}_{k}_{iter}", iter=iter)
    corr = results["correlation"][0]["correlation"]
    if wandb.run is not None:
        wandb.log({"correlation": corr, "iter": iter})
    return corr


def get_evaluation_fn(cfg, trainsets, valsets, experiment_folder):
    return partial(
        eval_fn,
        trainsets=trainsets,
        valsets=valsets,
        experiment_folder=experiment_folder,
    )
