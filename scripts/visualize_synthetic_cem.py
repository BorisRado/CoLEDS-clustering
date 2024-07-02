import pickle
from pathlib import Path

import numpy as np
import hydra
from tqdm import tqdm
from hydra.core.config_store import OmegaConf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.data.synthetic_dataset import generate_synthetic_datasets
from src.utils.stochasticity import set_seed
from src.clustering.clusterer import Clusterer
from src.cem.helper import load_cem
from src.data.partitioning import train_test_split


def draw_shape(ax, shape, xy, color, size=100):
    if shape == 'circle':
        ax.scatter(xy[0], xy[1], c=[color], s=size, marker='o')
    elif shape == 'square':
        ax.scatter(xy[0], xy[1], c=[color], s=size, marker='s')
    elif shape == 'triangle':
        ax.scatter(xy[0], xy[1], c=[color], s=size, marker='^')
    elif shape == 'cross':
        ax.scatter(xy[0], xy[1], c=[color], s=size, marker='x')



@hydra.main(version_base=None, config_path="../conf", config_name="visualize_synthetic")
def run(cfg):
    print(OmegaConf.to_yaml(cfg))
    folder = Path(cfg.folder)
    cem_config = OmegaConf.load(folder / "config.yaml")
    cem = load_cem(cem_config, folder)

    set_seed(cfg.general.seed)
    evalsets = generate_synthetic_datasets(
        cfg.final_evaluation.n_holdout_clients,
        cfg.final_evaluation.dataset_size,
        cfg.final_evaluation.p
    )

    clusterer = Clusterer(cem, {})

    shapes, colors, embeddings = [], [], []
    for ds in tqdm(evalsets):

        shape, color = ds[0]["shape"], ds[0]["color"]
        ds_ = train_test_split(ds, 0.001)[0]
        emb = clusterer.get_embedding(ds_)

        shapes.append(shape)
        colors.append(color)
        embeddings.append(emb.reshape(1, -1))

    embeddings = np.vstack(embeddings)
    if embeddings.shape[1] > 500:
        print("Applying PCA")
        embeddings = PCA(embeddings.shape[0]).fit_transform(embeddings)
        print("Applied PCA")
    xy_coordinates = TSNE(n_components=2, perplexity=20.).fit_transform(embeddings)

    with open(f'{folder}/visual.pkl', 'wb') as file:
        pickle.dump({
            "coords": xy_coordinates,
            "colors": colors,
            "shapes": shapes
        }, file)

if __name__ == "__main__":
    run()
