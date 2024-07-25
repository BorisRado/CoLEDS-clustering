from copy import deepcopy

import torch
from hydra.core.config_store import OmegaConf

from src.cem.cem import CEM, check_dtype
from src.utils.statistics import compute_statistic


class EmbeddingSpaceCEM(CEM):

    def __init__(self, model, reduction_stats):
        super().__init__()
        self.encoder = deepcopy(model["encoder"])
        if isinstance(reduction_stats, str):
            reduction_stats = [reduction_stats]
        else:
            reduction_stats = OmegaConf.to_container(reduction_stats)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.reduction_stats = reduction_stats

    @check_dtype
    def get_embedding(self, dataset):
        dataloader = self.get_dataloader(dataset)

        x = torch.vstack([
            self.encoder(b["img"].to(self.device)).flatten(start_dim=1) for b in dataloader
        ])
        assert x.ndim == 2 and x.shape[0] == len(dataloader.dataset)
        stats = compute_statistic(x, self.reduction_stats, 0).cpu().numpy()
        return stats

    def __str__(self):
        model_name = self.encoder.__class__.__name__
        return f"EmbeddingSpaceCEM_{model_name}"
