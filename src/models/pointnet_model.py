import numpy as np
import torch
import torch.nn as nn
from hydra.core.config_store import OmegaConf

from src.utils.statistics import compute_statistic
from src.models.models import PointNetEncoder


class PointNetModel(nn.Module):
    def __init__(self, reduction_stats, input_shape, head_sizes):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)

        if isinstance(reduction_stats, str):
            reduction_stats = [reduction_stats]
        else:
            reduction_stats = OmegaConf.to_container(reduction_stats)
        assert isinstance(reduction_stats, list)
        print("reduction statistics", reduction_stats, type(reduction_stats), isinstance(reduction_stats, OmegaConf))
        self.reduction_stats = reduction_stats

        self.head = self._get_head(input_shape, head_sizes)

    def _get_head(self, input_shape, head_sizes):
        tmp = self.encoder(torch.randn(tuple(input_shape)).unsqueeze(0))
        tmp = compute_statistic(tmp, self.reduction_stats, 0).shape[1]
        print(tmp)

        head_sizes = [tmp] + head_sizes
        head = []
        for i, o in zip(head_sizes[:-1], head_sizes[1:]):
            head.append(nn.Linear(i, o))
            head.append(nn.ReLU())
        return nn.Sequential(*head[:-1])

    def forward(self, dataset: torch.Tensor):
        x = self.encoder(dataset)
        x = compute_statistic(x, self.reduction_stats, 0)
        x = self.head(x)
        return x
