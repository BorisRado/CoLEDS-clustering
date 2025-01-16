import torch

from slwr.client.numpy_client import NumPyClient

from src.utils.parameters import get_parameters, set_parameters, get_gradients
from src.data.utils import get_dataloaders_with_replacement, get_label_distribution


class Client(NumPyClient):
    def __init__(self, model, trainset, valset, batch_size, num_client_updates):
        super().__init__()
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.batch_size = batch_size
        self.num_client_updates = num_client_updates
        val_distribution = get_label_distribution(valset, 10)
        self.val_dist_dict = {
            f"label{idx}": label for idx, label in enumerate(val_distribution.tolist())
        }
        self.val_set = torch.vstack([valset[i][0].unsqueeze(0) for i in range(len(valset))])

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.server_model_proxy.torch()
        self.model.train()
        self.model.zero_grad()
        set_parameters(self.model, parameters)

        trainloader = get_dataloaders_with_replacement(
            [self.trainset,],
            self.batch_size
        )[0]
        for _ in range(self.num_client_updates):
            dataloader_iter = iter(trainloader)
            batch1, batch2 = next(dataloader_iter), next(dataloader_iter)
            embs1, embs2 = self.model(batch1[0]), self.model(batch2[0])

            grad1, grad2 = self.server_model_proxy.serve_loss_computation_request(
                embeddings1=embs1,
                embeddings2=embs2,
            )
            embs1.backward(grad1)
            embs2.backward(grad2)

        return [g / self.num_client_updates for g in get_gradients(self.model)], 1, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(self.val_set)

        # issue: we cannot send a vector, so we need to encode
        # every element in a list to a entry in the output dictionary
        embedding_dict = {
            f"embedding{i}": emb for i, emb in enumerate(embedding.reshape(-1).tolist())
        }
        return 0.0, 1, embedding_dict | self.val_dist_dict
