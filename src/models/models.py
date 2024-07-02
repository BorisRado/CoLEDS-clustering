import torch.nn as nn


def _check_input_shape(input_shape):
    assert len(input_shape) == 3 and input_shape[1] == input_shape[2], str(input_shape)


class SimpleClfNet(nn.Sequential):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, input_shape, n_classes) -> None:
        _check_input_shape(input_shape)
        fc_dim = input_shape[1] // 4 - 3
        print(input_shape)
        super().__init__(
            nn.Conv2d(input_shape[0], 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * fc_dim * fc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )


class PointNetEncoder(nn.Sequential):
    def __init__(self, input_shape):
        _check_input_shape(input_shape)
        super().__init__(
            nn.Conv2d(input_shape[0], 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )


class SupervisedAutoencoder(nn.ModuleDict):
    def __init__(self, input_shape, n_classes):
        _check_input_shape(input_shape)
        encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=96,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )

        clf_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        tmp_padding = {
            28: 1,
            32: 0
        }[input_shape[1]]
        recon_head = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 3, 2, 1),
            nn.ConvTranspose2d(64, 32, 3, 2, 0),
            nn.ConvTranspose2d(32, 16, 3, 2, tmp_padding),
            nn.ConvTranspose2d(16, input_shape[0], 4, 2, 0),
        )
        super().__init__({
            "encoder": encoder,
            "clf_head": clf_head,
            "recon_head": recon_head
        })
