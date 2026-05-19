from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class SIGReg(nn.Module):
    """
    SIGReg loss from Balestriero and LeCun (2025).
    Code credit: https://github.com/galilai-group/lejepa
    """

    def __init__(
        self,
        n_knots: int = 17,
        t_max: float = 3.0,
        n_slices: int = 1024,
        n_patches: Optional[int] = None,
    ):
        """
        Args:
            n_knots: number of integration points for Epps-Pulley
            t_max: max integration bound
            n_slices: number of 1D slices to project embeddings onto
        """
        super().__init__()

        self.n_knots = n_knots
        self.t_max = t_max
        self.n_slices = n_slices
        self.n_patches = n_patches

        if n_knots < 2:
            raise ValueError("SIGReg requires n_knots >= 2")
        if n_slices < 1:
            raise ValueError("SIGReg requires n_slices >= 1")

        self.register_buffer("t", torch.empty(n_knots, dtype=torch.float32))
        self.register_buffer("phi", torch.empty(n_knots, dtype=torch.float32))
        self.register_buffer("weights", torch.empty(n_knots, dtype=torch.float32))
        self.init_weights()

    def init_weights(self) -> None:
        # LeJEPA paper approximates integral using trapezoid method
        t = torch.linspace(0, self.t_max, self.n_knots, dtype=torch.float32, device=self.t.device)
        dt = self.t_max / (self.n_knots - 1)
        weights = torch.full((self.n_knots,), 2 * dt, dtype=torch.float32, device=self.weights.device)
        weights[[0, -1]] = dt

        # Epps-Pulley downweighting term - CF of N(0, 1)
        w_t = torch.exp(-t.square() / 2.0)
        self.t.copy_(t)
        self.phi.copy_(w_t)  # set target CF to also be w_t
        self.weights.copy_(weights * w_t)

    def sample_patch_tokens(self, proj: Float[Tensor, "V ... D"]) -> Float[Tensor, "V ... D"]:
        if self.n_patches is None:
            return proj

        if proj.ndim < 4:
            raise ValueError(
                f"n_patches was set, but input does not look patch-shaped. "
                f"Expected [V, B, P, D], got {tuple(proj.shape)}"
            )

        P = proj.shape[-2]
        K = min(self.n_patches, P)
        idx = torch.randperm(P, device=proj.device)[:K]
        return proj.index_select(dim=-2, index=idx)

    def forward(self, proj: Float[Tensor, "V ... D"]) -> Float[Tensor, ""]:  # noqa: F722
        if self.n_patches is not None:
            proj = self.sample_patch_tokens(proj)

        samples = proj.reshape(-1, proj.shape[-1])  # [N, D]

        slices = torch.randn(samples.size(-1), self.n_slices, device=samples.device, dtype=samples.dtype)
        slices = slices.div_(slices.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12))

        x_t = (samples @ slices).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * samples.size(0)
        return statistic.mean()
