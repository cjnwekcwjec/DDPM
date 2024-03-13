import torch
import torch.nn.functional as F
from torch import nn
from utils import gather
from typing import Tuple, Optional


class Diffusion(nn.Module):
    def __init__(self, model, n_steps: int, device: torch.device):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.model = model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha = self.alpha ** 0.5
        self.sqrt_alpha_bar = self.alpha_bar ** 0.5
        self.one_alpha = 1 - self.alpha
        self.one_alpha_bar = 1 - self.alpha_bar

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        mean = gather(self.sqrt_alpha_bar, t) * x0
        var = gather(self.one_alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None):
        mean, var = self.q_xt_x0(x0, t)
        if eps is None:
            eps = torch.randn_like(x0)
        xt = mean + (var ** 0.5) * eps
        return xt

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None):
        coff = gather(self.one_alpha, t) / gather(self.one_alpha_bar ** 0.5, t)
        mean = (1 / gather(self.alpha ** 0.5, t)) * (xt - coff * self.model(xt, t))
        var = gather(self.beta, t)
        if eps is None:
            eps = torch.randn_like(xt)
        return mean + var ** 0.5 * eps

    def loss(self, x0: torch.Tensor, noise: torch.Tensor = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,)).to(self.device)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_theta = self.model(xt, t)
        return F.mse_loss(eps_theta, noise)

