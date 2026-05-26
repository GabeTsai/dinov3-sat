import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class JEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        global_views: Float[Tensor, "V ... D"], 
        local_views: Float[Tensor, "V ... D"] | None = None
    ) -> Float[Tensor, ""]:  # noqa: F722
        if local_views is not None:
            views = torch.cat([global_views, local_views], dim=0)
        else:
            views = global_views
        target = global_views.mean(0)
        return (target - views).square().mean()
        
