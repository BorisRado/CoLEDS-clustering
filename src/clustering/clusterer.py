import numpy as np
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans


class Clusterer:

    def __init__(self, profiler, datasets_dict):
        super().__init__()
        self.profiler = profiler
        self.init_embeddings = {
            key: np.vstack([self.get_embedding(ds) for ds in datasets])
            for key, datasets in datasets_dict.items()
        }

    def get_embedding(self, dataset):
        with torch.no_grad():
            emb = self.profiler.get_embedding(dataset)
        return preprocessing.normalize(emb)

    def init_kmeans_model(self, n_clusters, partition="train"):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        try:
            preds = self.kmeans.fit_predict(self.init_embeddings[partition])
        except Exception as e:
            print("Error in KMeans fit_predict", e)
            print("Trying to convert embeddings to double precision")
            preds = self.kmeans.fit_predict(self.init_embeddings[partition].astype(np.float64))
            print("Success with double precision")
        return preds

    def predict_client_cluster(self, client_dataset):
        embedding = self.get_embedding(client_dataset)
        cluster = self.kmeans.predict(embedding)
        assert isinstance(cluster, np.ndarray) and cluster.size == 1
        return cluster.item()

    def predict(self, partition):
        return self.kmeans.predict(self.init_embeddings[partition])
