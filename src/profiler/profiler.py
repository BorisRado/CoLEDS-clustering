from abc import ABC, abstractmethod

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


def check_dtype(func):
    def wrapper(self, dataset):
        assert isinstance(dataset, Dataset), f"{dataset} {type(dataset)}"
        assert not torch.is_grad_enabled()
        res = func(self, dataset)
        assert isinstance(res, np.ndarray) and res.shape[0] == 1 and res.ndim == 2
        return res
    return wrapper


class Profiler(ABC):

    @abstractmethod
    def get_embedding(self, dataset: Dataset) -> np.ndarray:
        """Get the embedding of a given dataset

        Parameters
        ----------
        data : Dataset
            Dataset to be used for getting the client embedding

        Returns
        -------
        np.ndarray
            Client embedding with shape (1,m)
        """
        return np.empty((1,0),)

    def get_dataloader(self, dataset, batch_size=32, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
