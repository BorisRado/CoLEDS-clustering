import torch


def test(model, testloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    forward_fn = lambda x: model["clf_head"](model["encoder"](x)) \
        if isinstance(model, torch.nn.ModuleDict) else model(x)
    correct = 0
    with torch.no_grad():
        for img, label in testloader:
            images = img.to(device)
            labels = label.to(device)
            outputs = forward_fn(images)
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
    criterion = torch.nn.MSELoss(reduction="sum")

    loss = 0.
    with torch.no_grad():
        for img, _ in testloader:
            images = img.to(device)

            outputs = model["recon_head"](model["encoder"](images))
            loss += criterion(outputs, images)

    model.to("cpu")
    return loss.item() / len(testloader.dataset)
