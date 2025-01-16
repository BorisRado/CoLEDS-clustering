from functools import reduce

import numpy as np
import torch

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from slwr.server.strategy import PlainSlStrategy

from src.models.helper import init_optimizer
from src.utils.parameters import get_parameters
from src.testing.correlation import compute_correlation


class Strategy(PlainSlStrategy):

    def __init__(self, model, optim_kwargs, evaluation_freq, **kwargs):
        super().__init__(**kwargs)
        self.losses = []
        self.model = model
        self.optimizer = init_optimizer(self.model.parameters(), **optim_kwargs)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)
        self.evaluation_freq = evaluation_freq

    def initialize_parameters(self, client_manager):
        _ = (client_manager,)
        return ndarrays_to_parameters(get_parameters(self.model))

    def configure_evaluate(self, server_round, parameters, client_manager):
        if (server_round - 1) % self.evaluation_freq == 0:
            return super().configure_evaluate(server_round, parameters, client_manager)
        return []

    def aggregate_server_fit(self, server_round, results):
        assert len(results) == 1
        result = results[0]
        round_loss = result.config["loss"]
        self.losses.append(round_loss)
        return []

    def aggregate_fit(self, server_round, results, failures):
        assert len(failures) == 0
        _ = (server_round,)
        self.optimizer.zero_grad()

        client_gradients = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        summed_gradients = [
            reduce(np.add, layer_gradients)
            for layer_gradients in zip(*client_gradients)
        ]
        for grad, param in zip(summed_gradients, self.
        model.parameters()):
            assert grad.shape == param.shape
            param.grad = torch.from_numpy(grad)
        self.optimizer.step()
        return ndarrays_to_parameters(get_parameters(self.model)), {}

    def aggregate_evaluate(self, server_round, results, failures):
        assert len(failures) == 0
        _ = (server_round, failures)
        evaluation_metrics = [res.metrics for _, res in results]
        metrics_keys = evaluation_metrics[0].keys()
        num_label_keys = sum(1 for k in metrics_keys if k.startswith("label"))
        num_emb_keys = sum(1 for k in metrics_keys if k.startswith("embedding"))

        embeddings, label_distributions = [], []
        for metric in evaluation_metrics:
            e = np.array([metric[f"embedding{i}"] for i in range(num_emb_keys)]).reshape(1, -1)
            l = np.array([metric[f"label{i}"] for i in range(num_label_keys)]).reshape(1, -1)
            embeddings.append(e)
            label_distributions.append(l)
        embeddings = np.vstack(embeddings)
        label_distributions = np.vstack(label_distributions)

        correlation = compute_correlation(label_distributions, embeddings)
        print("Current correlation", correlation)
        self.scheduler.step()
        return correlation, {}
