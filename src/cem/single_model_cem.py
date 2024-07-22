import warnings

import torch

from src.cem.cem import CEM, check_dtype


warnings.filterwarnings('ignore')


class SingleModelCEM(CEM):

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @check_dtype
    def get_embedding(self, dataset):
        dl = self.get_dataloader(dataset)
        x = torch.vstack([b["img"].to(self.device) for b in dl])
        emb = self.model(x)
        return emb.cpu().numpy()

    def __str__(self):
        model_name = self.model.__class__.__name__
        return f"SingleModelCEM_{model_name}"
