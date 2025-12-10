import torch.nn as nn
from flwr.client import NumPyClient
from torch.utils.data import DataLoader

from src.models.evaluation_procedures import test, test_ae
from src.models.helper import init_optimizer
from src.utils.parameters import get_parameters, set_parameters
from src.utils.stochasticity import set_seed


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, trainset, valset, model, batch_size, train_fn):
        super().__init__()
        set_seed()
        self.model = model
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        self.train_fn = train_fn

    def get_parameters(self, config):
        _ = (config,)
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        optimizer = init_optimizer(self.model.parameters(), **config)
        for _ in range(2):
            loss = self.train_fn(self.model, self.trainloader, optimizer=optimizer)
        return self.get_parameters({}), len(self.trainloader.dataset), {"loss": loss}

    def evaluate(self, parameters, config):
        _ = (config,)
        set_parameters(self.model, parameters)

        if isinstance(self.model, nn.ModuleDict) and "recon_head" in self.model:
            loss = test_ae(self.model, self.valloader)
            return loss, len(self.valloader.dataset), {"recon_loss": loss}

        accuracy = test(self.model, self.valloader)
        return accuracy, len(self.valloader.dataset), {"accuracy": accuracy}
