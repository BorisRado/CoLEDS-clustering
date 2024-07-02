import numpy as np
import torch
import torch.nn as nn

from src.models.models import PointNetEncoder


class TransformerEncoderModel(nn.Module):

    def __init__(self, input_shape, n_layers, **kwargs):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)

        self.head, self.init_embedding = self._get_head(input_shape, n_layers, **kwargs)

    def _get_head(self, input_shape, n_layers, **kwargs):

        tmp = self.encoder(torch.randn(tuple(input_shape)).unsqueeze(0)).shape[1]
        self.emb_dim = tmp
        init_embedding = nn.Parameter(torch.zeros(size=(1, tmp)))
        encoder_block = nn.TransformerEncoderLayer(d_model=tmp, batch_first=True, **kwargs)
        encoder = nn.TransformerEncoder(encoder_layer=encoder_block, num_layers=n_layers)
        return encoder, init_embedding

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = torch.cat(
            (self.init_embedding.repeat(x.shape[0], 1, 1), x), dim=1
        )
        x = self.head(x)
        x = x[:,0]
        assert x.shape == (1, self.emb_dim)
        return x.reshape(1, -1)
