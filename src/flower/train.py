from copy import deepcopy
from functools import partial

import torch
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

from src.flower.client import client_fn
from src.flower.strategy import get_strategy_with_chechpoint


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    out = {}
    ns = [num_examples for num_examples, _ in metrics]
    for k in metrics[0][1].keys():
        vals = [num_examples * m[k] for num_examples, m in metrics]
        out[k] = sum(vals) / sum(ns)
    return out


def train_flower(
    model,
    trainsets,
    valsets,
    batch_size,
    optim_kwargs,
    n_rounds,
    experiment_folder,
    train_fn,
    fraction_fit,
    model_save_name="fl_model.pth",
):
    client_fn_ = partial(
        client_fn,
        trainsets=trainsets,
        valsets=valsets,
        model=model,
        batch_size=batch_size,
        train_fn=train_fn
    )

    file = experiment_folder / model_save_name
    strategy_clz = get_strategy_with_chechpoint(FedAvg, file, deepcopy(model))
    strategy = strategy_clz(
        evaluation_frequency=1,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda *args, **kwargs: optim_kwargs
    )

    cr = {"num_cpus": 6}
    if torch.cuda.is_available():
        cr["num_gpus"] = 0.2
    history = start_simulation(
        client_fn=client_fn_,
        num_clients=len(trainsets),
        config=ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=cr
    )
    # history.losses_distributed
    model.load_state_dict(torch.load(file))
    return model
