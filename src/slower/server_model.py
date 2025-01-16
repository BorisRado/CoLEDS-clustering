import numpy as np
import torch
from slwr.server.server_model.numpy_server_model import NumPyServerModel

from src.models.losses import ContrastiveLoss
from slwr.server.server_model.utils import pytorch_format



class ServerModel(NumPyServerModel):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.criterion = ContrastiveLoss(2, temperature)
        self.criterion.mask = None
        self.losses = []

    @pytorch_format
    def serve_loss_computation_request(self, embeddings1, embeddings2):
        if self.criterion.mask is None:
            nc = embeddings1.shape[0]
            self.criterion.mask = (~torch.eye(nc * 2, nc * 2, dtype=bool)).float()

        embeddings1.requires_grad_(True)
        embeddings2.requires_grad_(True)
        loss = self.criterion(embeddings1, embeddings2)
        loss.backward()
        self.losses.append(loss.item())
        return [embeddings1.grad, embeddings2.grad]

    def get_fit_result(self):
        return [], {"loss": np.mean(self.losses).item()}
