import torch
import torch.nn as nn

from src.models.models import PointNetEncoder


class TransformerEncoderModel(nn.Module):

    def __init__(self, input_shape, token_dim, n_layers, **kwargs):
        super().__init__()
        self.dp_encoder = PointNetEncoder(input_shape=input_shape)
        self.token_dim = token_dim
        emb_dim = self.dp_encoder(torch.randn(tuple(input_shape)).unsqueeze(0)).shape[1]
        self.dp_encoder = nn.Sequential(self.dp_encoder, nn.Linear(emb_dim, token_dim))
        self.aggregator_model, self.cls_token = self._get_head(input_shape, n_layers, **kwargs)
        self.aggregator_model.layers[-1].self_attn

    def _get_head(self, input_shape, n_layers, **kwargs):
        cls_token = nn.Parameter(torch.zeros(size=(1, 1, self.token_dim)))
        encoder_block = nn.TransformerEncoderLayer(d_model=self.token_dim, batch_first=True, **kwargs)
        encoder = nn.TransformerEncoder(encoder_layer=encoder_block, num_layers=n_layers)
        return encoder, cls_token

    def forward(self, x: torch.Tensor):
        x = self.dp_encoder(x)
        x = x.reshape(1, x.shape[0], self.token_dim)
        x = torch.cat((self.cls_token, x), dim=1)
        x = self.aggregator_model(x)
        x = x[:,0]
        assert x.shape == (1, self.token_dim)
        return x
