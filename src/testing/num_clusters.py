import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_silhouette_scores(embeddings, max_clusters):
    start_time = time.time()
    silhouette_scores = []
    for k in range(2, min(max_clusters, embeddings.shape[0]-1)):  # Try different numbers of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        s = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(s)
    print(f"Silhouette computation time: {time.time() - start_time}")
    return {
        "scores": silhouette_scores,
        "nc": list(range(2, max_clusters))
    }
