import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class JEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, views: Float[Tensor, "V ... D"]) -> Float[Tensor, ""]:  # noqa: F722
        return (views.mean(0) - views).square().mean()
