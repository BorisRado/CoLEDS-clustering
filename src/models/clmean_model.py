import torch
import torch.nn as nn

from src.models.models import PointNetEncoder



class ClMeanModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder = PointNetEncoder(input_shape=input_shape)

    def forward(self, dataset: torch.Tensor):
        # Support both (batch_size, C, H, W) and (num_batches, batch_size, C, H, W) inputs
        if dataset.ndim == 4:
            # Standard case: (batch_size, C, H, W)
            x = self.encoder(dataset)
            return x.mean(axis=0, keepdim=True)
        elif dataset.ndim == 5:
            # Batch of batches: (num_batches, batch_size, C, H, W)
            num_batches, batch_size, c, h, w = dataset.shape
            # Reshape to (num_batches * batch_size, C, H, W)
            dataset_reshaped = dataset.view(num_batches * batch_size, c, h, w)
            # Encode all samples
            x = self.encoder(dataset_reshaped)
            # Get feature dimension from encoded output
            feature_dim = x.shape[1]
            # Reshape back to (num_batches, batch_size, feature_dim)
            x = x.view(num_batches, batch_size, feature_dim)
            # Compute mean for each batch: (num_batches, feature_dim)
            return x.mean(axis=1)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {dataset.ndim}D tensor with shape {dataset.shape}")


if __name__ == "__main__":
    # check that batched processing is equal to sequential processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10):
        batches = [torch.randn(2, 3, 32, 32, dtype=torch.double, device=device) for _ in range(10)]
        model =ClMeanModel((3, 32, 32)).to(device).double()
        seq = torch.vstack([model(b) for b in batches])
        batch = model(torch.vstack([b.unsqueeze(0) for b in batches]))
        assert (seq - batch == 0.).all()
    print("OK")
