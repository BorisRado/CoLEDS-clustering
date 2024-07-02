from copy import deepcopy

from flwr.client import NumPyClient
from torch.utils.data import DataLoader

from src.models.training_procedures import train_ce
from src.models.evaluation_procedures import test, test_ae
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
        for _ in range(2):
            self.train_fn(self.model, self.trainloader, **config)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        _ = (config,)
        set_parameters(self.model, parameters)

        accuracy = test(self.model, self.valloader)
        recon_loss = test_ae(self.model, self.valloader)
        return accuracy, len(self.valloader.dataset), \
            {"accuracy": accuracy, "recon_loss": recon_loss}


def client_fn(cid: str, trainsets, valsets, model, batch_size, train_fn):
    """Create and return an instance of Flower `Client`."""
    cid = int(cid)

    return FlowerClient(
        trainset=trainsets[cid],
        valset=valsets[cid],
        model=deepcopy(model),
        batch_size=batch_size,
        train_fn=train_fn
    ).to_client()
