import copy
from typing import List

import torch

from src.profiler.profiler import Profiler, check_dtype
from src.utils.statistics import compute_statistics


class EmbeddingSpaceProfiler(Profiler):

    def __init__(self, model: torch.nn.ModuleDict, statistics: List[str]):
        super().__init__()
        self.encoder = copy.deepcopy(model)
        self.statistics = statistics
        self.model_type = "autoencoder" if "recon_head" in model else "classifier"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @check_dtype
    def get_embedding(self, dataset):
        dataloader = self.get_dataloader(dataset, batch_size=64, shuffle=False)
        self.encoder.to(self.device)

        data = torch.vstack([
            self.encoder.get_embedding(b.to(self.device)).flatten(start_dim=1) for b, _ in dataloader
        ])

        stats = compute_statistics(data, self.statistics)
        return stats.reshape(1, -1)

    def __str__(self):
        return f"ES_{self.model_type}"
