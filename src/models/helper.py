import torch.optim as optim

def init_optimizer(model_parameters, optimizer_name="adam", **kwargs):
    optimizer = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    }[optimizer_name]

    return optimizer(model_parameters, **kwargs)
