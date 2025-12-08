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
        cls_token = nn.Parameter(torch.zeros(size=(1, 1, self.token_dim)), requires_grad=True)
        encoder_block = nn.TransformerEncoderLayer(d_model=self.token_dim, batch_first=True, **kwargs)
        encoder = nn.TransformerEncoder(encoder_layer=encoder_block, num_layers=n_layers)
        return encoder, cls_token

    def forward(self, x: torch.Tensor):
        # Support both (batch_size, C, H, W) and (num_batches, batch_size, C, H, W) inputs
        if x.ndim == 4:
            # Standard case: (batch_size, C, H, W)
            x = self.dp_encoder(x)
            x = x.reshape(1, x.shape[0], self.token_dim)
            x = torch.cat((self.cls_token, x), dim=1)
            x = self.aggregator_model(x)
            x = x[:,0]
            assert x.shape == (1, self.token_dim)
            return x
        elif x.ndim == 5:
            # Batch of batches: (num_batches, batch_size, C, H, W)
            num_batches, batch_size, c, h, w = x.shape
            # Reshape to (num_batches * batch_size, C, H, W)
            x_reshaped = x.view(num_batches * batch_size, c, h, w)
            # Encode all samples
            x_encoded = self.dp_encoder(x_reshaped)
            # Reshape to (num_batches, batch_size, token_dim)
            x_encoded = x_encoded.view(num_batches, batch_size, self.token_dim)
            # Process each batch separately through transformer
            outputs = []
            for i in range(num_batches):
                batch_x = x_encoded[i]  # (batch_size, token_dim)
                batch_x = batch_x.unsqueeze(0)  # (1, batch_size, token_dim)
                batch_x = torch.cat((self.cls_token, batch_x), dim=1)  # (1, batch_size+1, token_dim)
                batch_x = self.aggregator_model(batch_x)  # (1, batch_size+1, token_dim)
                batch_x = batch_x[:,0]  # (1, token_dim)
                outputs.append(batch_x)
            # Stack outputs: (num_batches, 1, token_dim) -> (num_batches, token_dim)
            return torch.cat(outputs, dim=0)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.ndim}D tensor with shape {x.shape}")

# NOTE: Don't use this model -- SL messes up batch normalization
