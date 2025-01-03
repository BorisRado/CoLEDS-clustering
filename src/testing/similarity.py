import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


np.seterr(all="raise")


def _compute_within_cluster_similarity(distributions):

    sim = cosine_similarity(distributions)
    similarities = sim[np.triu_indices(sim.shape[0], 1)]

    gt_similarity = np.mean(similarities)

    all_similarities = []
    for _ in range(100):
        sample = np.random.choice(similarities.shape[0], similarities.shape[0], replace=True)
        sim = np.mean(similarities[sample])
        all_similarities.append(sim)
    se = np.std(all_similarities) / 10
    return gt_similarity, se


def compute_similarity(clusterer, gt_distribution, max_clusters=30):

    similarities = []
    for n_clusters in range(1, max_clusters):
        clusterer.init_kmeans_model(n_clusters, "train")
        predicted_clusters = clusterer.predict("val")

        for cluster_idx in range(n_clusters):
            n_cluster_clients = (predicted_clusters == cluster_idx).sum()
            cluster_gt_distribution = gt_distribution[predicted_clusters == cluster_idx]

            if cluster_gt_distribution.shape[0] < 2:
                continue
            similarity, similarity_se = _compute_within_cluster_similarity(
                cluster_gt_distribution
            )
            similarities.append({
                "n_clusters": int(n_clusters),
                "n_cluster_clients": int(n_cluster_clients),
                "cluster_idx": int(cluster_idx),
                "similarity": float(similarity),
                "similarity_se": float(similarity_se)
            })
    return similarities
