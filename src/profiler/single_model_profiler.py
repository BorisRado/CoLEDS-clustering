import warnings

import torch
from torch.utils.data import TensorDataset

from src.profiler.profiler import Profiler, check_dtype


warnings.filterwarnings('ignore')


class SingleModelProfiler(Profiler):

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    @check_dtype
    def get_embedding(self, dataset):
        assert isinstance(dataset, TensorDataset)
        assert self.device == dataset.tensors[0].device
        x = dataset.tensors[0]
        emb = self.model(x)
        return emb.cpu().numpy()

    def __str__(self):
        model_name = self.model.__class__.__name__
        return f"SingleModelCEM_{model_name}"
