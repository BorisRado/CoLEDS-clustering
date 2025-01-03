from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_silhouette_scores(embeddings, max_clusters):
    silhouette_scores = []
    for k in range(2, max_clusters):  # Try different numbers of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        s = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(s)
    return {
        "scores": silhouette_scores,
        "nc": list(range(2, max_clusters))
    }
