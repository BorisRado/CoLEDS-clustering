import torch

from src.utils.parameters import get_gradients


class ContrastiveClient:
    def __init__(self, model, dataloader):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device, non_blocking=True)

    def get_embeddings(self):
        iterator = iter(self.dataloader)
        batch_1_imgs = next(iterator)[0]
        batch_2_imgs = next(iterator)[0]

        self.embedding_1 = self.model(batch_1_imgs)
        self.embedding_2 = self.model(batch_2_imgs)

        return self.embedding_1.detach(), self.embedding_2.detach()

    def backward(self, grad_1, grad_2):
        self.embedding_1.backward(grad_1)
        self.embedding_2.backward(grad_2)
        return get_gradients(self.model)

    def get_gradients(self):
        return get_gradients(self.model)
