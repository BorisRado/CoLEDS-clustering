import time
from copy import deepcopy
from random import sample

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.cl_client import ContrastiveClient
from src.models.losses import ContrastiveLoss
from src.models.helper import init_optimizer


def _optimization_iteration_sl(model, epoch_loaders, criterion, num_client_updates):

    epoch_clients = [ContrastiveClient(deepcopy(model), dl) for dl in epoch_loaders]

    losses = []
    for _ in range(num_client_updates):

        embeddings_1, embeddings_2 = zip(*[c.get_embeddings() for c in epoch_clients])
        embeddings_1, embeddings_2 = torch.vstack(embeddings_1), torch.vstack(embeddings_2)

        embeddings_1.requires_grad_(True)
        embeddings_2.requires_grad_(True)

        loss = criterion(embeddings_1, embeddings_2)
        loss.backward()
        losses.append(loss.detach().cpu().item())

        grad_1, grad_2 = embeddings_1.grad, embeddings_2.grad

        for i in range(len(epoch_clients)):
            epoch_clients[i].backward(grad_1[i].unsqueeze(0), grad_2[i].unsqueeze(0))

    gradient = None
    for i in range(len(epoch_clients)):
        client_gradient = epoch_clients[i].get_gradients()
        client_gradient = [grad / num_client_updates for grad in client_gradient]

        if gradient is None:
            gradient = client_gradient
        else:
            for i, grad in enumerate(client_gradient):
                gradient[i] += grad
    gradient = [g for g in gradient]

    # apply the gradient
    for grad, param in zip(gradient, model.parameters()):
        assert grad.shape == param.shape
        param.grad = grad

    return np.mean(losses)


def _optimized_gradient_computation(model, epoch_loaders, criterion, num_client_updates):
    losses = []
    for _ in range(num_client_updates):
        imgs1, imgs2 = [], []
        for el in epoch_loaders:
            il = iter(el)
            imgs1.append(next(il)[0].unsqueeze(0))
            imgs2.append(next(il)[0].unsqueeze(0))
        imgs1 = torch.vstack(imgs1)
        imgs2 = torch.vstack(imgs2)

        embs1, embs2 = model(imgs1), model(imgs2)
        # divide by num_client_updates since we are accumulating the gradient
        loss = criterion(embs1, embs2)
        # divide by num_client_updates as we are doing gradient accumulation
        (loss / num_client_updates).backward()
        losses.append(loss.detach().cpu().item())

    return np.mean(losses)


def train_contrastive(
    model,
    num_iterations,
    temperature,
    trainloaders,
    fraction_fit,
    optimizer,
    batch_size,
    optimized_computation,
    num_client_updates,
):
    start_time = time.time()
    print("Starting training")
    _ = (batch_size,) # we pass batch_size just because we pass all the arguments in hydra
    n_sampled_clients = int(len(trainloaders) * fraction_fit)

    criterion = ContrastiveLoss(n_sampled_clients, temperature)

    optimization_fn = _optimized_gradient_computation \
        if optimized_computation else _optimization_iteration_sl

    for idx in range(num_iterations):
        epoch_loaders = sample(trainloaders, k=n_sampled_clients)

        optimizer.zero_grad()
        loss = optimization_fn(
            model=model,
            epoch_loaders=epoch_loaders,
            criterion=criterion,
            num_client_updates=num_client_updates
        )
        optimizer.step()

        # bookkeeping
        if wandb.run is not None:
            wandb.log({"cl_loss": loss})
        if idx % 10 == 0:
            print(idx, loss)
    print(f"Contrastive iteration time: {time.time() - start_time:.2f}")


def train_ce(model, dataloader, proximal_mu=0.005, optimizer=None, **kwargs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if proximal_mu > 0:
        global_model = deepcopy(model)
        global_model.to(device)
    model.to(device)

    model.train()
    if optimizer is None:
        optimizer = init_optimizer(model.parameters(), **kwargs)

    criterion = nn.CrossEntropyLoss()
    losses = []
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

        losses.append(loss.detach().cpu().item())
    model.to("cpu")
    return np.mean(losses).item()


def train_vae(model, dataloader, beta=5e-4, optimizer=None, **kwargs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    if optimizer is None:
        optimizer = init_optimizer(model.parameters(), **kwargs)

    losses = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        recon, mu, logvar = model(imgs)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, imgs, reduction="sum")

        # KL Divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kld_loss
        loss /= imgs.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    model.to("cpu")
    return np.mean(losses).item()
