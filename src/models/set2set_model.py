import warnings

import torch
import torch.nn as nn

from src.models.models import PointNetEncoder


warnings.filterwarnings('ignore')
# UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters(). (Triggered internally at /opt/conda/conda-bld/pytorch_1716905969824/work/aten/src/ATen/native/cudnn/RNN.cpp:1424.)


class Set2Set(nn.Module):
    def __init__(self, input_dim, processing_steps, num_layers=1):
        super().__init__()
        D = input_dim
        self.input_dim = D
        self.output_dim = 2 * D
        self.processing_steps = processing_steps

        self.lstm = nn.LSTM(
            input_size=2 * D,
            hidden_size=D,
            num_layers=num_layers,
            batch_first=False,
        )
        self.lstm.reset_parameters()

    def forward(self, x):
        # x: [B, T, D] or [T, D]
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, _, D = x.shape  # B = dim_size

        # match PyG: q_star [B, 2D], h: ([num_layers, B, D], [num_layers, B, D])
        q_star = x.new_zeros((B, 2 * D))
        h = (
            x.new_zeros((self.lstm.num_layers, B, D)),
            x.new_zeros((self.lstm.num_layers, B, D)),
        )

        for _ in range(self.processing_steps):
            # input to LSTM: [seq_len=1, batch=B, input_size=2D]
            q, h = self.lstm(q_star.unsqueeze(0), h)   # q: [1, B, D]
            q = q.view(B, D)                           # [B, D]

            # attention over T
            e = (x * q.unsqueeze(1)).sum(dim=-1)       # [B, T]
            a = torch.softmax(e, dim=1)                # [B, T]
            r = torch.bmm(a.unsqueeze(1), x)           # [B, 1, D]
            r = r.squeeze(1)                           # [B, D]

            q_star = torch.cat([q, r], dim=-1)         # [B, 2D]

        out = q_star  # [B, 2D]
        return out


class Set2SetModel(nn.Module):
    def __init__(self, input_shape, processing_steps, hidden_dim, num_layers=1):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)
        emb_dim = self.encoder(torch.randn(tuple(input_shape)).unsqueeze(0)).shape[1]
        self.projection = nn.Linear(emb_dim, hidden_dim)
        self.rnn = Set2Set(input_dim=hidden_dim, processing_steps=processing_steps, num_layers=num_layers)

    def forward(self, dataset: torch.Tensor):
        # Support both (batch_size, C, H, W) and (num_batches, batch_size, C, H, W) inputs
        if dataset.ndim == 4:
            # Standard case: (batch_size, C, H, W)
            x = self.encoder(dataset)
            x = self.projection(x)
            x = self.rnn(x)
            return x
        elif dataset.ndim == 5:
            num_batches, batch_size, c, h, w = dataset.shape
            dataset_reshaped = dataset.view(num_batches * batch_size, c, h, w)
            x = self.encoder(dataset_reshaped)
            x = self.projection(x)
            x = x.view(num_batches, batch_size, -1)
            rnn_out = self.rnn(x)
            return rnn_out

        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {dataset.ndim}D tensor with shape {dataset.shape}")


if __name__ == "__main__":
    # check that batched processing is equal to sequential processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10):
        batches = [torch.randn(2, 3, 32, 32, dtype=torch.double, device=device) for _ in range(10)]
        model = Set2SetModel((3, 32, 32), processing_steps=5, hidden_dim=20).to(device).double()
        seq = torch.vstack([model(b) for b in batches])
        batch = model(torch.vstack([b.unsqueeze(0) for b in batches]))
        assert seq.allclose(batch)

    # make sure that custom implementation matches the one in torch_geometric
    from torch_geometric.nn import Set2Set as PyGSet2Set
    T = 10          # sequence length
    D = 16          # embedding dimension
    B = 8           # batch size
    STEPS = 3
    N_LAYERS = 1

    pyg_model = PyGSet2Set(D, processing_steps=STEPS, num_layers=N_LAYERS)
    my_model = Set2Set(D, processing_steps=STEPS, num_layers=N_LAYERS)

    my_model.lstm.load_state_dict(pyg_model.lstm.state_dict(), strict=True)

    batches = [torch.randn(T, D) for _ in range(B)]           # list of 8 sequences
    batched_tensor = torch.stack(batches, dim=0)              # [8, 10, 16]

    pyg_outputs = []
    for x in batches:
        out = pyg_model(x).squeeze()
        pyg_outputs.append(out)

    pyg_outputs = torch.stack(pyg_outputs, dim=0)  # [B, 2D]
    my_outputs = my_model(batched_tensor)  # [B, 2D]
    print("Max difference:", (pyg_outputs - my_outputs).abs().max().item())
    assert torch.allclose(pyg_outputs, my_outputs, atol=1e-6), "Mismatch!"

    print("Tests run, all ok!")
