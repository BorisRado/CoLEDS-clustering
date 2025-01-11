from copy import deepcopy
from random import sample

import wandb
import numpy as np
import torch
import torch.nn as nn

from src.utils.cl_client import ContrastiveClient
from src.models.losses import ContrastiveLoss
from src.models.helper import init_optimizer


def _optimization_iteration(criterion, client_list, num_client_updates):

    losses = []
    for _ in range(num_client_updates):

        embeddings_1, embeddings_2 = zip(*[c.get_embeddings() for c in client_list])
        embeddings_1, embeddings_2 = torch.vstack(embeddings_1), torch.vstack(embeddings_2)

        embeddings_1.requires_grad_(True)
        embeddings_2.requires_grad_(True)

        loss = criterion(embeddings_1, embeddings_2)
        loss.backward()
        losses.append(loss.detach().cpu().item())

        grad_1, grad_2 = embeddings_1.grad, embeddings_2.grad

        for i in range(len(client_list)):
            client_list[i].backward(grad_1[i].unsqueeze(0), grad_2[i].unsqueeze(0))

    gradient = None
    for i in range(len(client_list)):
        client_gradient = client_list[i].get_gradients()
        client_gradient = [grad / num_client_updates for grad in client_gradient]

        if gradient is None:
            gradient = client_gradient
        else:
            for i, grad in enumerate(client_gradient):
                gradient[i] += grad
    gradient = [g for g in gradient]
    return np.mean(losses), gradient


def train_contrastive(
    model,
    num_iterations,
    temperature,
    trainloaders,
    fraction_fit,
    optimizer,
    batch_size,
    num_client_updates,
):
    print("Starting training")
    n_sampled_clients = int(len(trainloaders) * fraction_fit)

    criterion = ContrastiveLoss(n_sampled_clients, temperature)
    for idx in range(num_iterations):
        epoch_loaders = sample(trainloaders, k=n_sampled_clients)
        epoch_clients = [ContrastiveClient(deepcopy(model), dl) for dl in epoch_loaders]

        optimizer.zero_grad()
        loss, gradient = _optimization_iteration(
            criterion=criterion,
            client_list=epoch_clients,
            num_client_updates=num_client_updates
        )

        if wandb.run is not None:
            wandb.log({"cl_loss": loss})
        if idx % 25 == 0:
            print(idx, loss)

        # apply the gradient
        for grad, param in zip(gradient, model.parameters()):
            assert grad.shape == param.shape
            param.grad = grad
        optimizer.step()


def train_ce(model, dataloader, proximal_mu=-1, optimizer=None, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(f"training on {device} {len(dataloader.dataset)}")

    if proximal_mu > 0:
        global_model = deepcopy(model)
        global_model.to(device)
    model.to(device)

    model.train()
    if optimizer is None:
        optimizer = init_optimizer(model.parameters(), **kwargs)

    criterion = nn.CrossEntropyLoss()
    for img, labels in dataloader:
        img, labels = img.to(device), labels.to(device)

        preds = model(img)
        loss = criterion(preds, labels)

        if proximal_mu > 0:
            proximal_loss = 0.0
            for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
                proximal_loss += torch.square((local_weights - global_weights).norm(2))
            loss += proximal_mu * proximal_loss

        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.to("cpu")
