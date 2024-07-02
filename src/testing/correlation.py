import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _all_equal(v):
    return (v == v[0]).all()


def compute_correlation(embeddings, gt_distributions):
    assert embeddings.shape[0] == gt_distributions.shape[0]
    assert isinstance(embeddings, np.ndarray) and isinstance(gt_distributions, np.ndarray)

    sim1 = cosine_similarity(embeddings)
    sim2 = cosine_similarity(gt_distributions)

    sim1 = sim1[np.triu_indices(sim1.shape[0], 1)]
    sim2 = sim2[np.triu_indices(sim2.shape[0], 1)]

    assert sim1.ndim == 1 and sim1.shape == sim2.shape
    gt_correlation = np.corrcoef(sim1, sim2)[0, 1]

    correlations = []
    for _ in range(100):
        sample = np.random.choice(sim1.shape[0], sim1.shape[0], replace=True)
        corr = np.corrcoef(sim1[sample], sim2[sample])[0, 1]
        correlations.append(corr)
    se = np.std(correlations) / np.sqrt(len(correlations))
    return [{"correlation": gt_correlation, "se": se}]
