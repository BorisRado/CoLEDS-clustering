from src.utils.stochasticity import TempRng
from src.utils.wandb import log_table
from src.testing.test_all import test_all


def eval_fn(cem, trainsets, valsets, n_classes, experiment_folder, iter, **kwargs):
    with TempRng(1602):
        results = test_all(cem, trainsets, valsets, n_classes, **kwargs)
    cem_name = cem.__class__.__name__
    for k, values in results.items():
        log_table(values, experiment_folder, f"{cem_name}_{k}_{iter}", iter=iter)
    return results["correlation"][0]["correlation"]
