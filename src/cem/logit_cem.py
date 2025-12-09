from copy import deepcopy

import torch

from src.cem.cem import CEM, check_dtype
from src.models.training_procedures import train_ce


class LogitCEM(CEM):

    def __init__(self, init_model, ft_epochs, batch_size, optim_kwargs, public_dataset=None):
        super().__init__()
        self.model = deepcopy(init_model)
        self.optim_kwargs = optim_kwargs
        self.ft_epochs = ft_epochs
        self.batch_size = batch_size
        assert self.public_dataset is not None
        self.public_dataset = public_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @check_dtype
    def get_embedding(self, dataset):
        dataloader = self.get_dataloader(dataset, self.batch_size, shuffle=True)
        model = deepcopy(self.model)
        with torch.enable_grad():
            for _ in range(self.ft_epochs):
                train_ce(model, dataloader, **self.optim_kwargs)

        del dataloader
        ho_dataset = self.public_dataset
        model.to(self.device)
        dataloader = self.get_dataloader(ho_dataset, shuffle=False)
        model.eval()
        out = torch.vstack([
            model(b.to(self.device)) for b, _ in dataloader
        ]).cpu().numpy()
        return out.reshape(1, -1)

    def __str__(self):
        model_name = self.model.__class__.__name__
        return f"LgC_{model_name}"
