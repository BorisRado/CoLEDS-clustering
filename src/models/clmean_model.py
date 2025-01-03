import torch
import torch.nn as nn

from src.models.models import PointNetEncoder



class ClMeanModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)

    def forward(self, dataset: torch.Tensor):
        x = self.encoder(dataset)
        return x.mean(axis=0, keepdim=True)
