import warnings

import torch
import torch.nn as nn

from src.models.models import PointNetEncoder
from torch_geometric.nn.aggr import Set2Set


warnings.filterwarnings('ignore')
# UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at /opt/conda/conda-bld/pytorch_1716905969824/work/aten/src/ATen/native/cudnn/RNN.cpp:1424.)

class Set2SetModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)
        emb_dim = self.encoder(torch.randn(tuple(input_shape)).unsqueeze(0)).shape[1]
        self.projection = nn.Linear(emb_dim, 48)
        self.rnn = Set2Set(in_channels=48, processing_steps=2, batch_first=True)

    def forward(self, dataset: torch.Tensor):
        x = self.encoder(dataset)
        x = self.projection(x)
        x = self.rnn(x)
        return x
