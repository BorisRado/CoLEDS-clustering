from PIL import Image

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as T
from datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def generate_random_shape_image(shape, color):
    fig, ax = plt.subplots(figsize=(3, 3))

    # Set up the figure
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")  # Turn off the axis

    # Draw the shape
    if shape == "circle":
        circle = plt.Circle([5, 5], 3, color=color)
        ax.add_patch(circle)
        label = 0
    elif shape == "square":
        square = plt.Rectangle([2, 2], 6, 6, color=color)
        ax.add_patch(square)
        label = 1
    elif shape == "triangle":
        triangle = plt.Polygon([[5, 8], [2, 2], [8, 2]], color=color)
        ax.add_patch(triangle)
        label = 2
    elif shape == 'cross':
        cross = plt.Polygon([[4, 10], [6, 10], [6, 6], [10, 6], [10, 4], [6, 4], [6, 0], [4, 0], [4, 4], [0, 4], [0, 6], [4, 6]], color=color)
        ax.add_patch(cross)
        label = 3

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)

    # Resize the array to (32, 32, 3) and transpose to (3, 32, 32)
    image = Image.fromarray(buf)
    image = image.resize((32, 32), resample=Image.Resampling.LANCZOS)

    return image, label

def get_color_variant(color_name):
    # List of possible variations: lighter, darker, etc.
    variations = [0.8, 1.2]

    # Get the base RGB color
    base_color = np.array(mcolors.to_rgb(color_name))

    # Randomly choose a variation
    variation = np.random.choice(variations)

    # Apply the variation to the base color
    variant_color = base_color * variation

    # Clip the values to be in the valid range [0, 1]
    variant_color += np.random.randn(3,) / 5
    variant_color = np.clip(variant_color, 0, 1)

    return variant_color

def generate_dataset(dataset_size, p, dictionary=True):
    tns = T.RandomAffine(45, (0., 0.5), (0.7, 1.3), 1., fill=1.)
    available_colors = ["red", "green", "blue", "yellow"]
    base_color = np.random.choice(available_colors)

    available_shapes = ["circle", "square", "triangle", "cross"]
    base_shape = np.random.choice(available_shapes)

    images, labels = [], []
    for _ in range(dataset_size):
        if np.random.uniform() < p:
            tmp, label = generate_random_shape_image(
                base_shape,
                get_color_variant(base_color)
            )
        else:
            tmp, label = generate_random_shape_image(
                np.random.choice(available_shapes),
                get_color_variant(np.random.choice(available_colors))
            )
        tmp = T.ToTensor()(tmp)[:3].unsqueeze(0)

        tmp = tns(tmp)
        labels.append(label)
        images.append(tmp)
    images = torch.vstack(images)

    if dictionary:
        dataset = Dataset.from_dict({
            "img": images,
            "shape": [base_shape] * images.shape[0],
            "color": [base_color] * images.shape[0],
            "label": labels
        })
        dataset.set_format("torch")
        return dataset
    else:
        labels = torch.tensor(labels)
        return TensorDataset(images, labels)

def generate_synthetic_datasets(n_datasets, dataset_size, p=0.8):
    datasets = []
    for _ in range(n_datasets):
        ds = generate_dataset(dataset_size, p)
        datasets.append(ds)
    return datasets
