import pickle

import numpy as np
import hydra
from hydra.core.config_store import OmegaConf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.data.synthetic_dataset import generate_synthetic_dataset
from src.utils.stochasticity import set_seed
from src.clustering.clusterer import Clusterer
from src.profiler.helper import load_profiler


@hydra.main(version_base=None, config_path="../../conf", config_name="visualize_synthetic")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    profiler, folder = load_profiler(cfg)

    set_seed(cfg.general.seed)
    clusterer = Clusterer(profiler, {})
    shapes, colors, embeddings = [], [], []
    for idx in range(cfg.final_evaluation.n_holdout_clients):
        # clearer, idiomatic
        if idx % 10 == 0: print(f"Processing holdout client {idx}")
        dataset = generate_synthetic_dataset(
            cfg.final_evaluation.dataset_size,
            cfg.final_evaluation.p,
            image_resolution=(32, 32)
        )

        shape, color = dataset.base_shape, dataset.base_color
        emb = clusterer.get_embedding(dataset)

        shapes.append(shape)
        colors.append(color)
        embeddings.append(emb.reshape(1, -1))

    embeddings = np.vstack(embeddings)
    if embeddings.shape[1] > 500:
        # reduce dimensionality a bit otherwise TSNE is too complex
        embeddings = PCA(embeddings.shape[0]).fit_transform(embeddings)
    xy_coordinates = TSNE(n_components=2, perplexity=20.).fit_transform(embeddings)

    with open(f"{folder}/visual.pkl", "wb") as file:
        pickle.dump({
            "coords": xy_coordinates,
            "colors": colors,
            "shapes": shapes
        }, file)


if __name__ == "__main__":
    run()
