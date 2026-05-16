import torch

from dinov3.utils.metrics import effective_rank, pairwise_cos_mean, std_mean, std_min


def test_metrics_accept_bfloat16_inputs():
    x = torch.randn(8, 4, dtype=torch.bfloat16)

    outputs = [
        std_mean(x),
        std_min(x),
        pairwise_cos_mean(x),
        effective_rank(x),
    ]

    for output in outputs:
        assert output.ndim == 0
        assert output.dtype == torch.float32
        assert torch.isfinite(output)


def test_effective_rank_single_sample_bfloat16():
    output = effective_rank(torch.ones(1, 4, dtype=torch.bfloat16))

    assert output.item() == 1.0
