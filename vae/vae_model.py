import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_size, z_dim, device, hidden_size=20):
        super(VAE, self).__init__()
        self.in_size = in_size
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.device = device

        self.enc = nn.Sequential(
            nn.Linear(in_size * in_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus()
        )
        self.z_mean = nn.Linear(hidden_size, z_dim)
        self.z_var = nn.Sequential(
            nn.Linear(hidden_size, z_dim),
            nn.Softplus()
        )

        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, in_size * in_size),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        x = self.enc(x)
        mean = self.z_mean(x)
        var = self.z_var(x)

        return mean, var

    def decoder(self, z):
        y = self.dec(z)

        return y

    def sample(self, mean, var):
        # backpropを使えるようにする
        # epsilen ~ N(0, 1)
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + epsilon * torch.exp(0.5 * var)

    def forward(self, x):
        x = x.view(-1, self.in_size * self.in_size)
        mean, var = self.encoder(x)
        z = self.sample(mean, var)
        y = self.decoder(z)

        return z, y

    def loss_sigmoid(self, x):
        x = x.view(-1, self.in_size * self.in_size)
        mean, var = self.encoder(x)
        delta = 1e-8
        KL = 0.5 * torch.sum(1 + var - mean**2 - torch.exp(var))
        z = self.sample(mean, var)
        y = self.decoder(z)
        reconstruction = torch.mean(
            x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta))
        lower_bound = [KL, reconstruction]
        return -sum(lower_bound)
