import copy

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans


class Clusterer:

    def __init__(self, cem, datasets_dict):
        super().__init__()
        self.cem = cem
        self.init_embeddings = {}
        for key, datasets in datasets_dict.items():
            self.init_embeddings[key] = np.vstack([
                self.get_embedding(ds) for ds in datasets
            ])
            print(f"Initial embeddings shape [{key}]: {self.init_embeddings[key].shape}")

    def get_embedding(self, dataset):

        # if self.cem.__class__.__name__ not in {"LabelCEM", "WeightDiffCEM"}:
        #     dataset = copy.deepcopy(dataset)
        #     dataset = dataset.remove_columns("label")

        with torch.no_grad():
            raw_emb = self.cem.get_embedding(dataset)
        return preprocessing.normalize(raw_emb)

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
