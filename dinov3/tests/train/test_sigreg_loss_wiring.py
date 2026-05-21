import torch
from omegaconf import OmegaConf
from torch import nn

import dinov3.train.ssl_meta_arch as ssl_meta_arch
from dinov3.layers.ffn_layers import Mlp
from dinov3.loss import SIGReg
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.train import get_effective_ibot_mask_probability


class _DINOLossStub:
    def __call__(self, **kwargs):
        return kwargs["student_logits"].sum() * 0


class _SIGRegStub:
    def __init__(self, n_patches=None):
        self.seen_shape = None
        self.seen_shapes = []
        self.n_patches = n_patches

    def __call__(self, views):
        self.seen_shape = tuple(views.shape)
        self.seen_shapes.append(self.seen_shape)
        return views.sum() * 0 + 2.0

    def sample_patch_tokens(self, views, n_patches=None):
        n_patches = self.n_patches if n_patches is None else n_patches
        if n_patches is None:
            return views
        return views[:, :, :n_patches]


class _ProjectionStub(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        return x[..., : self.out_dim]


class _RecordingSIGReg(SIGReg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_shapes = []

    def forward(self, views):
        self.seen_shapes.append(tuple(views.shape))
        return super().forward(views)


def _build_arch_for_compute_losses():
    arch = SSLMetaArch.__new__(SSLMetaArch)
    nn.Module.__init__(arch)
    arch.dino_loss = _DINOLossStub()
    arch.cfg = OmegaConf.create({"dino": {"reweight_dino_local_loss": False}})
    arch.dino_loss_weight = 1.0
    arch.dino_global_ignore_diagonal = True
    arch.dino_koleo_loss_weight = 0.0
    arch.ibot_loss_weight = 0.0
    arch.koleo_enabled = False
    arch.ibot_enabled = False
    arch.sigreg_on_cls = False
    arch.sigreg_on_patch = False
    arch.sigreg_cls_loss_weight = 0.0
    arch.sigreg_patch_loss_weight = 0.0
    arch.gram_use_loss = False
    arch.student = nn.ModuleDict()
    return arch


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def init_weights(self):
        pass


class _TinyHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_dim = kwargs["in_dim"]
        out_dim = kwargs["out_dim"]
        self.proj = nn.Linear(in_dim, out_dim)

    def init_weights(self):
        pass


class _TinyLoss:
    def __init__(self, *args, **kwargs):
        pass


def _minimal_ssl_cfg(cls_loss_weight=0.0, patch_loss_weight=0.0, use_proj=False):
    return OmegaConf.create(
        {
            "compute_precision": {"sharding_strategy": "SHARD_GRAD_OP"},
            "train": {"centering": "sinkhorn_knopp"},
            "crops": {"local_crops_number": 2},
            "dino": {
                "loss_weight": 1.0,
                "global_ignore_diagonal": True,
                "head_n_prototypes": 5,
                "head_bottleneck_dim": 4,
                "head_norm_last_layer": False,
                "head_nlayers": 2,
                "head_hidden_dim": 8,
                "koleo_loss_weight": 0.0,
                "koleo_loss_distributed": False,
                "koleo_topk": 1,
                "reweight_dino_local_loss": False,
            },
            "ibot": {
                "loss_weight": 0.0,
                "mask_sample_probability": 0.1,
                "mask_ratio_min_max": [0.1, 0.2],
                "separate_head": True,
                "head_n_prototypes": 5,
                "head_bottleneck_dim": 4,
                "head_norm_last_layer": False,
                "head_nlayers": 2,
                "head_hidden_dim": 8,
            },
            "sigreg": {
                "cls_loss_weight": cls_loss_weight,
                "patch_loss_weight": patch_loss_weight,
                "use_proj": use_proj,
                "proj_dim": 256,
                "n_knots": 3,
                "t_max": 1.0,
                "n_slices": 4,
                "n_patches": None,
            },
            "distillation": {"enabled": False},
            "gram": {"use_loss": False},
            "student": {"arch": "tiny"},
        }
    )


def _loss_inputs(global_patches=7, local_patches=7, dim=4):
    student_global = {
        "cls_after_head": torch.zeros(2, 3, 5, requires_grad=True),
        "cls_pre_head": torch.ones(2, 3, dim, requires_grad=True),
        "patch_pre_head": torch.ones(2, 3, global_patches, dim, requires_grad=True),
    }
    student_local = {
        "cls_after_head": torch.zeros(4, 3, 5, requires_grad=True),
        "cls_pre_head": torch.ones(4, 3, dim, requires_grad=True),
        "patch_pre_head": torch.ones(4, 3, local_patches, dim, requires_grad=True),
    }
    teacher_global = {"cls_centered": torch.zeros(2, 3, 5)}
    return dict(
        teacher_global=teacher_global,
        student_global=student_global,
        student_local=student_local,
        gram_global={},
        masks=torch.zeros(6, 7, dtype=torch.bool),
        mask_indices_list=torch.empty(0, dtype=torch.long),
        masks_weight=torch.empty(0),
        iteration=0,
    )


def test_sigreg_proj_modules_created_only_for_enabled_sigreg_losses(monkeypatch):
    def build_model_from_cfg(cfg, only_teacher=False):
        if only_teacher:
            return _TinyBackbone(), 4
        return _TinyBackbone(), _TinyBackbone(), 4

    monkeypatch.setattr(ssl_meta_arch, "build_model_from_cfg", build_model_from_cfg)
    monkeypatch.setattr(ssl_meta_arch, "DINOHead", _TinyHead)
    monkeypatch.setattr(ssl_meta_arch, "DINOLoss", _TinyLoss)
    monkeypatch.setattr(ssl_meta_arch, "iBOTPatchLoss", _TinyLoss)

    cls_only = SSLMetaArch(_minimal_ssl_cfg(cls_loss_weight=0.2, patch_loss_weight=0.0, use_proj=True))
    assert isinstance(cls_only.student["sigreg_cls_proj"], Mlp)
    assert cls_only.student["sigreg_cls_proj"].fc2.out_features == 256
    assert "sigreg_patch_proj" not in cls_only.student

    patch_only = SSLMetaArch(_minimal_ssl_cfg(cls_loss_weight=0.0, patch_loss_weight=0.2, use_proj=True))
    assert "sigreg_cls_proj" not in patch_only.student
    assert isinstance(patch_only.student["sigreg_patch_proj"], Mlp)
    assert patch_only.student["sigreg_patch_proj"].fc2.out_features == 256

    disabled = SSLMetaArch(_minimal_ssl_cfg(cls_loss_weight=0.0, patch_loss_weight=0.0, use_proj=True))
    assert "sigreg_cls_proj" not in disabled.student
    assert "sigreg_patch_proj" not in disabled.student


def test_zero_koleo_ibot_loss():
    arch = _build_arch_for_compute_losses()

    loss, loss_dict = arch.compute_losses(**_loss_inputs())

    assert torch.is_tensor(loss)
    assert "koleo_loss" not in loss_dict
    assert "ibot_loss" not in loss_dict


def test_sigreg_uses_student_global_local_cls_pre_head_views():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_cls = True
    arch.sigreg_cls_loss_weight = 0.25
    arch.sigreg_cls_loss = _SIGRegStub()

    loss, loss_dict = arch.compute_losses(**_loss_inputs())

    assert arch.sigreg_cls_loss.seen_shape == (6, 3, 4)
    assert loss_dict["sigreg_cls_loss"].item() == 2.0
    assert loss_dict["sigreg_cls_loss_weight"].item() == 0.25
    assert loss.item() == 0.5
    assert "sigreg_cls_std_mean" in loss_dict
    assert "sigreg_cls_std_min" in loss_dict
    assert "sigreg_cls_pairwise_cos_mean" in loss_dict
    assert "sigreg_cls_effective_rank" in loss_dict


def test_sigreg_expensive_metrics_are_rate_limited():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_cls = True
    arch.sigreg_cls_loss_weight = 0.25
    arch.sigreg_cls_loss = _SIGRegStub()
    inputs = _loss_inputs()
    inputs["iteration"] = 1

    _, loss_dict = arch.compute_losses(**inputs)

    assert "sigreg_cls_std_mean" in loss_dict
    assert "sigreg_cls_std_min" in loss_dict
    assert "sigreg_cls_pairwise_cos_mean" not in loss_dict
    assert "sigreg_cls_effective_rank" not in loss_dict


def test_sigreg_cls_projection_is_used_for_loss_and_metrics():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_cls = True
    arch.sigreg_cls_loss_weight = 0.25
    arch.sigreg_cls_loss = _SIGRegStub()
    arch.student["sigreg_cls_proj"] = _ProjectionStub(out_dim=2)

    _, loss_dict = arch.compute_losses(**_loss_inputs())

    assert arch.sigreg_cls_loss.seen_shape == (6, 3, 2)
    assert torch.isfinite(loss_dict["sigreg_cls_std_mean"])
    assert torch.isfinite(loss_dict["sigreg_cls_effective_rank"])


def test_sigreg_patch_samples_to_common_patch_count():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_patch = True
    arch.sigreg_patch_loss_weight = 0.125
    arch.sigreg_patch_loss = _SIGRegStub()
    inputs = _loss_inputs()
    inputs["iteration"] = 1

    loss, loss_dict = arch.compute_losses(**inputs)

    assert arch.sigreg_patch_loss.seen_shapes == [(4, 3, 7, 4), (2, 3, 7, 4)]
    assert loss_dict["sigreg_patch_local_loss"].item() == 2.0
    assert loss_dict["sigreg_patch_global_loss"].item() == 2.0
    assert loss_dict["sigreg_patch_local_loss_weight"].item() == 0.125
    assert loss_dict["sigreg_patch_global_loss_weight"].item() == 0.125
    assert loss.item() == 0.25
    assert "sigreg_patch_local_std_mean" in loss_dict
    assert "sigreg_patch_global_std_mean" in loss_dict


def test_sigreg_patch_projection_is_used_after_patch_sampling():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_patch = True
    arch.sigreg_patch_loss_weight = 0.125
    arch.sigreg_patch_loss = _SIGRegStub(n_patches=2)
    arch.student["sigreg_patch_proj"] = _ProjectionStub(out_dim=3)

    _, loss_dict = arch.compute_losses(**_loss_inputs(global_patches=7, local_patches=5, dim=4))

    assert arch.sigreg_patch_loss.seen_shapes == [(4, 3, 2, 3), (2, 3, 2, 3)]
    assert torch.isfinite(loss_dict["sigreg_patch_local_std_mean"])
    assert torch.isfinite(loss_dict["sigreg_patch_global_effective_rank"])


def test_sigreg_patch_expensive_metrics_use_flattened_samples():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_patch = True
    arch.sigreg_patch_loss_weight = 0.125
    arch.sigreg_patch_loss = _SIGRegStub()

    _, loss_dict = arch.compute_losses(**_loss_inputs())

    assert arch.sigreg_patch_loss.seen_shapes == [(4, 3, 7, 4), (2, 3, 7, 4)]
    assert "sigreg_patch_local_pairwise_cos_mean" in loss_dict
    assert "sigreg_patch_global_pairwise_cos_mean" in loss_dict


def test_sigreg_patch_metrics_respect_patch_sampling():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_patch = True
    arch.sigreg_patch_loss_weight = 0.125
    arch.sigreg_patch_loss = _SIGRegStub(n_patches=2)

    _, loss_dict = arch.compute_losses(**_loss_inputs())

    assert arch.sigreg_patch_loss.seen_shapes == [(4, 3, 2, 4), (2, 3, 2, 4)]
    assert torch.isfinite(loss_dict["sigreg_patch_local_std_mean"])
    assert torch.isfinite(loss_dict["sigreg_patch_global_pairwise_cos_mean"])


def test_sigreg_patch_computes_with_global_local_patch_grid_mismatch():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_on_patch = True
    arch.sigreg_patch_loss_weight = 0.125
    sigreg_patch_loss = _RecordingSIGReg(n_knots=3, t_max=1.0, n_slices=4, n_patches=64)
    object.__setattr__(arch, "sigreg_patch_loss", sigreg_patch_loss)
    inputs = _loss_inputs(global_patches=256, local_patches=49, dim=8)
    inputs["iteration"] = 1

    loss, loss_dict = arch.compute_losses(**inputs)

    assert sigreg_patch_loss.seen_shapes == [(4, 3, 49, 8), (2, 3, 49, 8)]
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_dict["sigreg_patch_local_loss"])
    assert torch.isfinite(loss_dict["sigreg_patch_global_loss"])


def test_zero_ibot_weight_disables_masking_unless_forced():
    cfg = OmegaConf.create(
        {
            "ibot": {
                "loss_weight": 0.0,
                "mask_sample_probability": 0.5,
                "force_masking": False,
            }
        }
    )
    assert get_effective_ibot_mask_probability(cfg) == 0.0

    cfg.ibot.force_masking = True
    assert get_effective_ibot_mask_probability(cfg) == 0.5
