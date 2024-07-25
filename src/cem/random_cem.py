import torch

from src.cem.cem import CEM, check_dtype


class RandomCEM(CEM):

    def __init__(self, dimension, n_classes):
        _ = (n_classes,)
        self.dimension = dimension

    @check_dtype
    def get_embedding(self, dataset):
        _ = (dataset,)
        return torch.randn(1, self.dimension).numpy()

    def __str__(self):
        return "RandomCEM"
