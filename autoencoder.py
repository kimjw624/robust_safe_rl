from pathlib import Path

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Configurable MLP autoencoder.

    Example:
        model = AutoEncoder(
            input_dim=48,
            hidden_dims=(128, 64),
            latent_dim=3,
            activation="relu",
        )
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64),
        latent_dim=3,
        activation="relu",
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.activation_name = activation

        act = self._activation(activation)

        encoder_layers = []
        last_dim = self.input_dim

        for h in self.hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(act())
            last_dim = h

        encoder_layers.append(nn.Linear(last_dim, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        last_dim = self.latent_dim

        for h in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h))
            decoder_layers.append(act())
            last_dim = h

        decoder_layers.append(nn.Linear(last_dim, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def _activation(name):
        name = name.lower()

        if name == "relu":
            return nn.ReLU

        if name == "tanh":
            return nn.Tanh

        if name == "elu":
            return nn.ELU

        if name == "gelu":
            return nn.GELU

        raise ValueError(f"Unsupported activation: {name}")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_error(self, x, reduction="none"):
        x_hat, z = self.forward(x)

        mse_per_sample = torch.mean((x_hat - x) ** 2, dim=1)

        if reduction == "mean":
            return mse_per_sample.mean(), z

        if reduction == "sum":
            return mse_per_sample.sum(), z

        return mse_per_sample, z

    def save(self, path, mean=None, std=None, thresholds=None, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "activation": self.activation_name,
            "mean": None if mean is None else mean.detach().cpu(),
            "std": None if std is None else std.detach().cpu(),
            "thresholds": {} if thresholds is None else thresholds,
            "metadata": {} if metadata is None else metadata,
        }

        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)

        model = cls(
            input_dim=payload["input_dim"],
            hidden_dims=payload["hidden_dims"],
            latent_dim=payload["latent_dim"],
            activation=payload.get("activation", "relu"),
        )

        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        model.eval()

        mean = payload.get("mean")
        std = payload.get("std")
        thresholds = payload.get("thresholds", {})

        metadata = payload.get("metadata", {})

        return model, mean, std, thresholds, metadata