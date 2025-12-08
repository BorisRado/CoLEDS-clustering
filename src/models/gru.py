import warnings

import torch
import torch.nn as nn

from src.models.models import PointNetEncoder


warnings.filterwarnings('ignore')
# UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at /opt/conda/conda-bld/pytorch_1716905969824/work/aten/src/ATen/native/cudnn/RNN.cpp:1424.)

class GRUModel(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)
        emb_dim = self.encoder(torch.randn(tuple(input_shape)).unsqueeze(0)).shape[1]
        self.projection = nn.Linear(emb_dim, hidden_dim)
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, dataset: torch.Tensor):
        # Support both (batch_size, C, H, W) and (num_batches, batch_size, C, H, W) inputs
        if dataset.ndim == 4:
            # Standard case: (batch_size, C, H, W)
            x = self.encoder(dataset)
            x = self.projection(x)
            x = x.unsqueeze(0)  # Add sequence dimension: (batch_size, 1, 48)
            rnn_out, _ = self.rnn(x)
            return rnn_out[:, -1, :]  # Return final embedding: (batch_size, 64)
        elif dataset.ndim == 5:
            # Batch of batches: (num_batches, batch_size, C, H, W)
            num_batches, batch_size, c, h, w = dataset.shape
            # Reshape to (num_batches * batch_size, C, H, W) for encoding
            dataset_reshaped = dataset.view(num_batches * batch_size, c, h, w)
            # Encode all samples
            x = self.encoder(dataset_reshaped)
            # Project
            x = self.projection(x)
            # Reshape to (num_batches, batch_size, 48) - this becomes (batch_size=num_batches, seq_len=batch_size, features=48) for RNN
            x = x.view(num_batches, batch_size, -1)
            # GRU with batch_first=True: (num_batches, batch_size, 48) -> (num_batches, batch_size, 64)
            rnn_out, _ = self.rnn(x)
            # Return final timestep for each batch: (num_batches, 64)
            return rnn_out[:, -1, :]
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {dataset.ndim}D tensor with shape {dataset.shape}")


if __name__ == "__main__":
    # check that batched processing is equal to sequential processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10):
        batches = [torch.randn(2, 3, 32, 32, dtype=torch.double, device=device) for _ in range(10)]
        model = GRUModel((3, 32, 32)).to(device).double()
        seq = torch.vstack([model(b) for b in batches])
        batch = model(torch.vstack([b.unsqueeze(0) for b in batches]))
        assert seq.allclose(batch, rtol=1e-10, atol=1e-15)
    print("Tests run, all ok!")
