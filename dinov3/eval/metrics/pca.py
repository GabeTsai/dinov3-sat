import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dinov3.checkpointer.checkpointer import distributed_checkpoint_to_state_dict
from dinov3.models import build_model_from_cfg
from PIL import Image
from omegaconf import OmegaConf
from dinov3.configs.config import get_default_config
from torchvision.transforms import v2

NUM_COMPONENTS = 3
RESIZE_SIZE = 1024

def make_transform_sar(resize_size: int) -> v2.Compose:
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.199,),
        std=(0.144,),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def load_teacher_backbone(checkpoint_path: str, cfg):
    """
    Load the EMA teacher backbone from a DCP training checkpoint.
    """
    backbone, _ = build_model_from_cfg(cfg, only_teacher=True)

    full_state = distributed_checkpoint_to_state_dict(checkpoint_path)["model"]
    teacher_backbone_state = {
        k.removeprefix("teacher.backbone."): v
        for k, v in full_state.items()
        if k.startswith("teacher.backbone.")
    }

    # assign=True to replace meta param objects with actual tensors in memory
    backbone.load_state_dict(teacher_backbone_state, strict=True, assign=True)
    backbone.eval()
    return backbone


def compute_pca(model, image_path, resize_size: int = 512, patch_size: int = 16):
    """
    Compute and visualise a 3-component PCA of the teacher patch features.
    """
    h_patches, w_patches = RESIZE_SIZE // patch_size, RESIZE_SIZE // patch_size
    image = Image.open(image_path)
    transform = make_transform_sar(resize_size=RESIZE_SIZE)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(image_tensor, is_training=True)

    patch_tokens = out["x_norm_patchtokens"][0].cpu().numpy()

    pca = PCA(n_components=NUM_COMPONENTS, whiten=True)
    projected = pca.fit_transform(patch_tokens)            
    projected = projected.reshape(h_patches, w_patches, NUM_COMPONENTS) 

    # sigmoid to map into [0, 1]
    projected = torch.sigmoid(torch.tensor(projected)).numpy()
    return projected

"""
Example usage: 
python3 -m dinov3.eval.metrics.pca \
    --checkpoint checkpoints/baselines/66999/ckpt \
    --output-dir checkpoints/baselines/66999/results \
    --config dinov3/configs/train/vitl_sar.yaml \
    --image test-images/noto-earthquake_00000067_post_disaster.tif
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--config",required=True, help="Training YAML config (e.g. vitl_sar.yaml).")
    parser.add_argument("--image",required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.merge(
        get_default_config(),
        OmegaConf.load(args.config),
    )

    patch_size = cfg.student.patch_size  # 16 for ViT-L SAR model

    model = load_teacher_backbone(args.checkpoint, cfg)
    pca_map = compute_pca(model, args.image, resize_size=RESIZE_SIZE, patch_size=patch_size)

    h, w = pca_map.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Image.open(args.image), cmap="gray")
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[1].imshow(pca_map, interpolation="bicubic")
    axes[1].set_title(f"PCA of patch tokens ({h}x{w} patches, {NUM_COMPONENTS} components)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_output.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()
