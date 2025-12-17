import os
from copy import deepcopy

import ray
import torch
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

from src.flower.client import FlowerClient
from src.flower.strategy import get_strategy_with_chechpoint
from src.utils.stochasticity import set_seed



def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    out = {}
    ns = [num_examples for num_examples, _ in metrics]
    for k in metrics[0][1].keys():
        vals = [num_examples * m[k] for num_examples, m in metrics]
        out[k] = sum(vals) / sum(ns)
    return out


def add_defaults_to_strategy_kwargs(kwargs):
    kwargs.setdefault("fraction_fit", 0.1)
    kwargs.setdefault("fraction_evaluate", 0.2)
    kwargs.setdefault("evaluation_frequency", 5)
    kwargs.setdefault("min_fit_clients", 1)
    kwargs.setdefault("min_evaluate_clients", 1)
    kwargs.setdefault("min_available_clients", 1)
    kwargs.setdefault("evaluate_metrics_aggregation_fn", weighted_average)
    kwargs.setdefault("fit_metrics_aggregation_fn", weighted_average)
    print("Strategy kwargs:")
    print(kwargs)


def train_flower(
    model,
    client_fn_kwargs,
    optim_kwargs,
    n_rounds,
    experiment_folder,
    strategy_kwargs,
    seed,
    model_save_name="fl_model.pth",
    model_return_strategy="latest",
):
    assert set(client_fn_kwargs.keys()) == {"trainsets", "valsets", "batch_size", "train_fn"}
    assert model_return_strategy in {"latest", "best_accuracy"}

    trainsets = client_fn_kwargs.pop("trainsets")
    valsets = client_fn_kwargs.pop("valsets")
    batch_size = client_fn_kwargs["batch_size"]
    train_fn = client_fn_kwargs["train_fn"]

    if not ray.is_initialized():
        ray.init()

    trainsets_refs = [ray.put(ts) for ts in trainsets]
    valsets_refs = [ray.put(ts) for ts in valsets]

    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        cid = int(cid)
        ts = ray.get(trainsets_refs[cid])
        vs = ray.get(valsets_refs[cid])

        return FlowerClient(
            trainset=ts,
            valset=vs,
            model=deepcopy(model),
            batch_size=batch_size,
            train_fn=train_fn
        ).to_client()

    file = str(experiment_folder / model_save_name)
    strategy_clz = get_strategy_with_chechpoint(FedAvg, file, deepcopy(model))

    add_defaults_to_strategy_kwargs(strategy_kwargs)
    strategy = strategy_clz(
        **strategy_kwargs,
        on_fit_config_fn=lambda *args, **kwargs: optim_kwargs
    )

    cr = {"num_cpus": 4}
    if torch.cuda.is_available():
        cr["num_gpus"] = 0.5

    set_seed(seed)
    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(trainsets),
        config=ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=cr,
        keep_initialised=True
    )
    # history.losses_distributed
    if model_return_strategy == "latest":
        weights = torch.load(file, weights_only=True)
    else:
        base, ext = os.path.splitext(file)
        best_file = f"{base}_best{ext}"
        weights = torch.load(best_file, weights_only=True)
    model.load_state_dict(weights)
    return model, history
