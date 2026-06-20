import torch
from PIL import Image

from dinov3.data.augmentations import DataAugmentationDINO
from dinov3.data.collate import collate_data_and_cast
from dinov3.data.transforms import AddLogGammaSpeckle


def test_add_log_gamma_speckle_preserves_shape_and_large_l_variance():
    torch.manual_seed(0)
    looks = 4096
    sigma_pix = 0.02
    image = torch.full((1, 512, 512), 0.5)
    transform = AddLogGammaSpeckle(
        p=1.0,
        sigma_pix_range=(sigma_pix, sigma_pix),
        looks_choices=(looks,),
        looks_probs=(1.0,),
    )

    out = transform(image)
    added_noise = out - image
    strength = sigma_pix / torch.sqrt(torch.special.polygamma(1, torch.tensor(float(looks))))
    expected_variance = strength.square() / looks

    assert out.shape == image.shape
    assert torch.isclose(added_noise.var(), expected_variance, rtol=0.01)


def test_gamma_speckle_augmentation_produces_teacher_crops_for_collation():
    image = Image.fromarray(torch.full((64, 64), 128, dtype=torch.uint8).numpy(), mode="L")
    transform = DataAugmentationDINO(
        global_crops_scale=(1.0, 1.0),
        local_crops_scale=(1.0, 1.0),
        local_crops_number=2,
        global_crops_size=32,
        local_crops_size=16,
        gaussian_blur=False,
        mean=(0.0,),
        std=(1.0,),
        gamma_speckle={
            "enabled": True,
            "student_p": 1.0,
            "student_sigma_pix": (0.08, 0.08),
            "teacher_p": 1.0,
            "teacher_sigma_pix": (0.02, 0.02),
            "looks_choices": (4096,),
            "looks_probs": (1.0,),
        },
    )

    torch.manual_seed(0)
    output = transform(image)
    batch = collate_data_and_cast(
        [(output, ())],
        mask_ratio_tuple=(0.0, 0.0),
        mask_probability=0.0,
        dtype=torch.float32,
        n_tokens=4,
        mask_generator=lambda _: torch.zeros(2, 2, dtype=torch.bool),
    )

    assert batch["collated_global_crops"].shape == (2, 1, 32, 32)
    assert batch["collated_global_crops_teacher"].shape == (2, 1, 32, 32)
    assert not torch.equal(batch["collated_global_crops"], batch["collated_global_crops_teacher"])
