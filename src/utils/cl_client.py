from src.models.helper import init_optimizer
from src.utils.parameters import get_gradients


class ContrastiveClient:
    def __init__(self, model, dataloader):
        super().__init__()
        self.model = model
        self.data_iterator = iter(dataloader)

    def get_embeddings(self):
        batch_1_imgs = next(self.data_iterator)["img"]
        batch_2_imgs = next(self.data_iterator)["img"]

        self.embedding_1 = self.model(batch_1_imgs)
        self.embedding_2 = self.model(batch_2_imgs)

        return self.embedding_1.detach(), self.embedding_2.detach()

    def backward(self, grad_1, grad_2):
        self.embedding_1.backward(grad_1)
        self.embedding_2.backward(grad_2)
        return get_gradients(self.model)
