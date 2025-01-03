import warnings

import torch

from src.cem.cem import CEM, check_dtype


warnings.filterwarnings('ignore')


class SingleModelCEM(CEM):

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    @check_dtype
    def get_embedding(self, dataset):
        dl = self.get_dataloader(dataset)
        x = torch.vstack([b[0] for b in dl]).to(self.device)
        with torch.no_grad():
            emb = self.model(x)
        return emb.cpu().numpy()

    def __str__(self):
        model_name = self.model.__class__.__name__
        return f"SingleModelCEM_{model_name}"
