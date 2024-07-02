import numpy as np
from torch.utils.data import Dataset


from src.cem.cem import CEM, check_dtype
from src.data.utils import get_label_distribution



class LabelCEM(CEM):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    @check_dtype
    def get_embedding(self, dataset: Dataset) -> np.ndarray:
        res = get_label_distribution(dataset, self.n_classes).reshape(1, -1)
        assert res.shape == (1, self.n_classes)
        assert res.sum() == len(dataset)
        return res
