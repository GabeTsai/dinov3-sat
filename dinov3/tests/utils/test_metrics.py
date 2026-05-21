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


def test_pairwise_cos_mean_matches_full_pairwise_matrix():
    x = torch.randn(16, 5)
    normalized = torch.nn.functional.normalize(x, dim=-1)
    sim = normalized @ normalized.T
    mask = ~torch.eye(x.shape[0], dtype=torch.bool)
    expected = sim[mask].mean()

    assert torch.allclose(pairwise_cos_mean(x), expected, atol=1e-6)


def test_effective_rank_matches_svd_formulation():
    x = torch.randn(32, 6)
    normalized = torch.nn.functional.normalize(x, dim=-1)
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    eigvals = torch.linalg.svdvals(centered).square()
    p = eigvals / eigvals.sum()
    expected = torch.exp(-(p * torch.log(p + 1e-12)).sum())

    assert torch.allclose(effective_rank(x), expected, atol=1e-5)
