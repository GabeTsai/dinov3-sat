# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
from typing import Callable, Dict, Tuple, Union
import torch
from torch import Tensor, nn

logger = logging.getLogger("dinov3")


def convert_patch_embed_channels(
    state_dict: Dict[str, torch.Tensor],
    target_in_chans: int,
    proj_key: str = "backbone.patch_embed.proj.weight",
) -> Dict[str, torch.Tensor]:
    """Convert patch_embed.proj.weight in a CPU state dict to a different number of input channels.

    Must be called before tensors are distributed (e.g. as DTensors) since it changes the weight shape.
    For target_in_chans == 1: sums across the source channels (equivalent to patch_first_conv pretrained=True).
    For target_in_chans > src_chans: tiles source channels then scales by src/target ratio.
    """
    if proj_key not in state_dict:
        return state_dict
    w = state_dict[proj_key]
    src_chans = w.shape[1]
    if src_chans == target_in_chans:
        return state_dict
    if target_in_chans == 1:
        new_w = w.sum(dim=1, keepdim=True)
    else:
        indices = torch.arange(target_in_chans) % src_chans
        new_w = w[:, indices] * (src_chans / target_in_chans)
    logger.info(f"Converted {proj_key} from {src_chans} -> {target_in_chans} channels")
    return {**state_dict, proj_key: new_w}


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
