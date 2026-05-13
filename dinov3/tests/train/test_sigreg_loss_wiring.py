import torch
from omegaconf import OmegaConf

from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.train import get_effective_ibot_mask_probability


class _DINOLossStub:
    def __call__(self, **kwargs):
        return kwargs["student_logits"].sum() * 0


class _SIGRegStub:
    def __init__(self):
        self.seen_shape = None

    def __call__(self, views):
        self.seen_shape = tuple(views.shape)
        return views.sum() * 0 + 2.0


def _build_arch_for_compute_losses():
    arch = SSLMetaArch.__new__(SSLMetaArch)
    arch.dino_loss = _DINOLossStub()
    arch.cfg = OmegaConf.create({"dino": {"reweight_dino_local_loss": False}})
    arch.dino_loss_weight = 1.0
    arch.dino_global_ignore_diagonal = True
    arch.dino_koleo_loss_weight = 0.0
    arch.ibot_loss_weight = 0.0
    arch.koleo_enabled = False
    arch.ibot_enabled = False
    arch.sigreg_use_loss = False
    arch.sigreg_loss_weight = 0.0
    arch.gram_use_loss = False
    return arch


def _loss_inputs():
    student_global = {
        "cls_after_head": torch.zeros(2, 3, 5, requires_grad=True),
        "cls_pre_head": torch.ones(2, 3, 4, requires_grad=True),
    }
    student_local = {
        "cls_after_head": torch.zeros(4, 3, 5, requires_grad=True),
        "cls_pre_head": torch.ones(4, 3, 4, requires_grad=True),
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


def test_zero_koleo_ibot_loss():
    arch = _build_arch_for_compute_losses()

    loss, loss_dict = arch.compute_losses(**_loss_inputs())

    assert torch.is_tensor(loss)
    assert "koleo_loss" not in loss_dict
    assert "ibot_loss" not in loss_dict


def test_sigreg_uses_student_global_local_cls_pre_head_views():
    arch = _build_arch_for_compute_losses()
    arch.sigreg_use_loss = True
    arch.sigreg_loss_weight = 0.25
    arch.sigreg_loss = _SIGRegStub()

    loss, loss_dict = arch.compute_losses(**_loss_inputs())

    assert arch.sigreg_loss.seen_shape == (6, 3, 4)
    assert loss_dict["sigreg_loss"].item() == 2.0
    assert loss_dict["sigreg_loss_weight"].item() == 0.25
    assert loss.item() == 0.5


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
