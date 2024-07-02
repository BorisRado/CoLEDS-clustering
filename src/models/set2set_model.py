import torch
import torch.nn as nn

from src.models.models import PointNetEncoder
from torch_geometric.nn.aggr import Set2Set


class Set2SetModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)
        emb_dim = self.encoder(torch.randn(tuple(input_shape)).unsqueeze(0)).shape[1]
        self.projection = nn.Linear(emb_dim, 48)
        self.rnn = Set2Set(in_channels=48, processing_steps=5, batch_first=True)

    def forward(self, dataset: torch.Tensor):
        x = self.encoder(dataset)
        x = self.projection(x)
        x = self.rnn(x)
        return x
