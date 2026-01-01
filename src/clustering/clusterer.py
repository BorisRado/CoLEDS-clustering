from functools import partial

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering


class Clusterer:

    def __init__(self, profiler, datasets_dict, algorithm="kmeans"):
        super().__init__()
        assert algorithm in {
            "kmeans", "hierarchical_ward", "hierarchical_complete", "hierarchical_average"
        }
        if algorithm == "kmeans":
            self.clustering_cls = partial(KMeans, random_state=10)
        else:
            assert algorithm.startswith("hierarchical_")
            linkage = algorithm.split("_")[1]
            self.clustering_cls = partial(AgglomerativeClustering, linkage=linkage)

        self.profiler = profiler
        self.algorithm = algorithm
        self.init_embeddings = {
            key: np.vstack([self.get_embedding(ds) for ds in datasets])
            for key, datasets in datasets_dict.items()
        }

    def get_embedding(self, dataset):
        with torch.no_grad():
            emb = self.profiler.get_embedding(dataset)
        return preprocessing.normalize(emb)

    def init_kmeans_model(self, n_clusters):
        assert "train" in self.init_embeddings
        self.kmeans = self.clustering_cls(n_clusters=n_clusters)
        preds = self.kmeans.fit_predict(self.init_embeddings["train"])
        return preds

    def predict_client_cluster(self, client_dataset):
        self.assert_kmeans()
        embedding = self.get_embedding(client_dataset)
        cluster = self.kmeans.predict(embedding)
        assert isinstance(cluster, np.ndarray) and cluster.size == 1
        return cluster.item()

    def predict(self, partition):
        self.assert_kmeans()
        return self.kmeans.predict(self.init_embeddings[partition])

    def assert_kmeans(self):
        assert self.algorithm == "kmeans", \
            f"You cannot call predict_client_cluster if not using the KMeans algorithm. Current algo: {self.clustering_algorithm}"
