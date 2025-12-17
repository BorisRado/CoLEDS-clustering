import torch
from torch.utils.data import DataLoader


def test(model, testloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for img, label in testloader:
            images = img.to(device)
            labels = label.to(device)
            outputs = model(images)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    model.to("cpu")
    return accuracy


def test_ae(model, testloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if not isinstance(model, torch.nn.ModuleDict):
        return -1

    recon_loss = 0.
    with torch.no_grad():
        for img, _ in testloader:
            images = img.to(device)
            recon, _, _ = model(images)
            recon_loss += torch.nn.functional.mse_loss(recon, images, reduction="sum")

    model.to("cpu")
    model.train()

    # Return average reconstruction loss (or total loss if you prefer)
    return recon_loss.item() / len(testloader.dataset)


def get_clustering_accuracy(models_dict, clusterer, datasets):
    out = {"dataset_size": [], "accuracy": [], "cluster_idx": [], "client_idx": []}

    for dataset in datasets:
        pred_cluster = clusterer.predict_client_cluster(dataset)

        dataloader = DataLoader(dataset, batch_size=128)

        acc = test(models_dict[pred_cluster], dataloader)

        out["dataset_size"].append(len(dataset))
        out["accuracy"].append(acc)
        out["cluster_idx"].append(pred_cluster)
        if hasattr(dataset, "_client_idx"):
            out["client_idx"].append(dataset._client_idx)
        else:
            out["client_idx"].append(-1)
    return out


def get_all_client_accuracy(model, datasets):
    out = {"dataset_size": [], "accuracy": [], "client_idx": []}
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=128)
        acc = test(model, dataloader)

        out["dataset_size"].append(len(dataset))
        out["accuracy"].append(acc)
        if hasattr(dataset, "_client_idx"):
            out["client_idx"].append(dataset._client_idx)
        else:
            out["client_idx"].append(-1)
    return out
