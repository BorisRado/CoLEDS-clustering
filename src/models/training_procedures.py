from copy import deepcopy
from random import sample

import wandb
import torch
import torch.nn as nn
from flwr.server.strategy.aggregate import aggregate

from src.utils.parameters import get_parameters, set_parameters, get_gradients
from src.utils.cl_client import ContrastiveClient
from src.models.losses import ContrastiveLoss
from src.models.helper import init_optimizer


def _optimization_iteration(criterion, client_list):

    embeddings_1, embeddings_2 = zip(*[client.get_embeddings() for client in client_list])
    embeddings_1, embeddings_2 = torch.vstack(embeddings_1), torch.vstack(embeddings_2)

    embeddings_1.requires_grad_(True)
    embeddings_2.requires_grad_(True)

    loss = criterion(embeddings_1, embeddings_2)
    loss.backward()

    gradient, grad_1, grad_2 = None, embeddings_1.grad, embeddings_2.grad

    for i in range(len(client_list)):
        client_gradient = client_list[i].backward(grad_1[i].unsqueeze(0), grad_2[i].unsqueeze(0))
        if gradient is None:
            gradient = client_gradient
        else:
            for i, grad in enumerate(client_gradient):
                gradient[i] += grad
    return loss.detach().cpu().item(), gradient


def train_contrastive_centralized(model, n_iterations, temperature, dataloaders, fraction_fit, **kwargs):
    n_sampled_clients = int(len(dataloaders) * fraction_fit)
    criterion = ContrastiveLoss(n_sampled_clients, temperature)
    optimizer = init_optimizer(model.parameters(), **kwargs)

    for idx in range(n_iterations):
        epoch_loaders = sample(dataloaders, k=n_sampled_clients)
        epoch_iter_loaders = [iter(dl) for dl in epoch_loaders]

        embeddings = []
        for _ in range(2):
            batches = [next(dl)["img"] for dl in epoch_iter_loaders]
            embs = torch.vstack([model(b) for b in batches])
            embeddings.append(embs)
        optimizer.zero_grad()
        loss = criterion(embeddings[0], embeddings[1])
        if wandb.run is not None:
            wandb.log({"cl_loss": loss.detach().cpu().item()})
        print(idx, loss)
        loss.backward()
        optimizer.step()


def train_contrastive(model, n_iterations, temperature, dataloaders, fraction_fit, optimizer):
    # train_contrastive_centralized(
    #     model=model,
    #     iterations=iterations,
    #     temperature=temperature,
    #     dataloaders=dataloaders,
    #     fraction_fit=fraction_fit,
    #     **kwargs
    # )

    n_sampled_clients = int(len(dataloaders) * fraction_fit)

    criterion = ContrastiveLoss(n_sampled_clients, temperature)
    for idx in range(n_iterations):
        epoch_loaders = sample(dataloaders, k=n_sampled_clients)
        epoch_clients = [ContrastiveClient(deepcopy(model), dl) for dl in epoch_loaders]

        optimizer.zero_grad()
        loss, gradient = _optimization_iteration(criterion, epoch_clients)

        if wandb.run is not None:
            wandb.log({"cl_loss": loss})
        print(idx, loss)

        # apply the gradient
        for grad, param in zip(gradient, model.parameters()):
            assert grad.shape == param.shape
            param.grad = grad
        optimizer.step()


def train_ce(model, dataloader, proximal_mu=-1, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(f"training on {device} {len(dataloader.dataset)}")

    if proximal_mu > 0:
        global_model = deepcopy(model)
        global_model.to(device)
    model.to(device)

    model.train()
    optimizer = init_optimizer(model.parameters(), **kwargs)

    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k in {"label", "img"}}

        preds = model(batch["img"])
        # print(preds)
        # print(batch["label"])
        loss = criterion(preds, batch["label"])

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

def train_supervised_autoencoder(model, dataloader, ae_weight, proximal_mu=-1, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"training on {device} {len(dataloader.dataset)}")

    if proximal_mu > 0:
        global_model = deepcopy(model)
        global_model.to(device)
    model.train()
    model.to(device)

    optimizer = init_optimizer(model.parameters(), **kwargs)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k in {"label", "img"}}

        emb = model["encoder"](batch["img"])

        loss = 0.
        if ae_weight > 0:
            img_preds = model["recon_head"](emb)
            loss += ae_weight * mse(img_preds, batch["img"])
        if ae_weight < 1.:
            cls_preds = model["clf_head"](emb)
            loss += (1-ae_weight) * ce(cls_preds, batch["label"])

        if proximal_mu > 0:
            proximal_loss = 0.0
            for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
                proximal_loss += torch.square((local_weights - global_weights).norm(2))
            loss += proximal_mu * proximal_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.to("cpu")
