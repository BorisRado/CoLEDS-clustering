import torch


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
