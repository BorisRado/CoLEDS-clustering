import numpy as np
from torch.utils.data import Dataset


from src.profiler.profiler import Profiler, check_dtype
from src.data.utils import get_label_distribution



class LabelProfiler(Profiler):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    @check_dtype
    def get_embedding(self, dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, "_label_distribution"):
            res = dataset._label_distribution.reshape(1, -1).detach().cpu().numpy()
        else:
            res = get_label_distribution(dataset, self.n_classes).reshape(1, -1)
        assert res.shape == (1, self.n_classes)
        assert res.sum() == len(dataset)
        return res

    def __str__(self):
        return "LabelProfiler"
