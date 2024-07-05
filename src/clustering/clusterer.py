import copy

import ray
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans


def _get_embedding(cem, dataset):
    with torch.no_grad():
        emb = cem.get_embedding(dataset)
    return preprocessing.normalize(emb)


@ray.remote(num_cpus=4, num_gpus=0.25 if torch.cuda.is_available() else 0.0)
class ModelActor:
    def __init__(self, cem):
        self.cem = cem

    def get_embedding(self, dataset):
        return _get_embedding(self.cem, dataset)


class Clusterer:

    def __init__(self, cem, datasets_dict):
        super().__init__()
        self.cem = cem
        self.init_embeddings = {}

        if not ray.is_initialized():
            ray.init()
        num_actors=4
        actors = [ModelActor.remote(copy.deepcopy(cem)) for _ in range(num_actors)]
        for key, datasets in datasets_dict.items():
            futures = [actors[i % num_actors].get_embedding.remote(ds)
                       for i, ds in enumerate(datasets)]
            self.init_embeddings[key] = np.vstack(ray.get(futures))
            print(f"Initial embeddings shape [{key}]: {self.init_embeddings[key].shape}")

    def get_embedding(self, dataset):
        return _get_embedding(self.cem, dataset)

    def init_kmeans_model(self, n_clusters, partition="train"):

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        preds = self.kmeans.fit_predict(self.init_embeddings[partition])
        return preds

    def predict_client_cluster(self, client_dataset):
        embedding = self.get_embedding(client_dataset)
        cluster = self.kmeans.predict(embedding)
        assert isinstance(cluster, np.ndarray) and cluster.size == 1
        return cluster.item()

    def predict(self, partition):
        return self.kmeans.predict(self.init_embeddings[partition])
