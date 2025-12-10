from copy import deepcopy

import numpy as np
import torch

from src.profiler.profiler import Profiler, check_dtype
from src.utils.parameters import get_parameters
from src.models.training_procedures import train_ce


def _flatten_model(model):
    params = get_parameters(model)
    return np.hstack([p.reshape(-1) for p in params])


class WeightDiffProfiler(Profiler):

    def __init__(self, init_model, ft_epochs, batch_size, optim_kwargs):
        super().__init__()
        self.model = deepcopy(init_model)
        self.batch_size = batch_size
        self.optim_kwargs = optim_kwargs
        self.ft_epochs = ft_epochs

    @check_dtype
    def get_embedding(self, dataset):
        model = deepcopy(self.model)
        dataloader = self.get_dataloader(dataset, batch_size=self.batch_size, shuffle=True)

        with torch.enable_grad():
            for _ in range(self.ft_epochs):
                train_ce(model, dataloader, **self.optim_kwargs)

        wd = _flatten_model(model) - _flatten_model(self.model)
        return wd.reshape(1, -1)

    def __str__(self):
        model_name = self.model.__class__.__name__
        return f"WDC_{model_name}"
