import torch
import torch.nn as nn


def _check_input_shape(input_shape):
    assert len(input_shape) == 3 and input_shape[1] == input_shape[2], str(input_shape)


class SimpleClfNet(nn.ModuleDict):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, input_shape, n_classes) -> None:
        _check_input_shape(input_shape)
        fc_dim = input_shape[1] // 4 - 3
        super().__init__({
            "encoder": nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(start_dim=1),
                nn.Linear(64 * fc_dim * fc_dim, 512),
            ),
            "clf_head": nn.Sequential(
                nn.ReLU(),
                nn.Linear(512, n_classes)
            )
        })

    def forward(self, x):
        return self["clf_head"](self["encoder"](x))

    def get_embedding(self, x):
        return self["encoder"](x)



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


class VariationalAutoencoder(nn.ModuleDict):
    """Variational Autoencoder with convolutional encoder and decoder"""

    def __init__(self, input_shape, latent_dim=15) -> None:
        _check_input_shape(input_shape)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        fc_dim = input_shape[1] // 4 - 3
        self.fc_dim = fc_dim

        super().__init__({
            "encoder": nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(start_dim=1),
            ),
            "fc_mu": nn.Linear(64 * fc_dim * fc_dim, latent_dim),
            "fc_logvar": nn.Linear(64 * fc_dim * fc_dim, latent_dim),
            "recon_head": nn.Sequential(
                nn.Linear(latent_dim, 64 * fc_dim * fc_dim),
                nn.ReLU(),
                nn.Unflatten(1, (64, fc_dim, fc_dim)),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ConvTranspose2d(64, 32, 5, stride=1, padding=0),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ConvTranspose2d(32, input_shape[0], 5, stride=1, padding=0),
            )
        })

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self["encoder"](x)
        mu = self["fc_mu"](h)
        logvar = self["fc_logvar"](h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self["recon_head"](z)

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_embedding(self, x):
        """Get latent embedding (mean) for input"""
        mu, _ = self.encode(x)
        return mu
