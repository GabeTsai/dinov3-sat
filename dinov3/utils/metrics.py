import torch
import torch.nn.functional as F


@torch.no_grad()
def std_mean(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Average per-dimension std across batch.

    Args:
        x: feature tensor of shape [B, D]
        normalize: whether to L2-normalize features first

    Returns:
        scalar tensor
    """
    if normalize:
        x = F.normalize(x.float(), dim=-1)

    std_per_dim = x.std(dim=0, unbiased=False)  # [D]
    return std_per_dim.mean()


@torch.no_grad()
def std_min(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Minimum per-dimension std across batch.

    Args:
        x: feature tensor of shape [B, D]
        normalize: whether to L2-normalize features first

    Returns:
        scalar tensor
    """
    if normalize:
        x = F.normalize(x.float(), dim=-1)

    std_per_dim = x.std(dim=0, unbiased=False)  # [D]
    return std_per_dim.min()


@torch.no_grad()
def pairwise_cos_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Mean off-diagonal pairwise cosine similarity.

    Args:
        x: feature tensor of shape [B, D]

    Returns:
        scalar tensor
    """
    B = x.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=x.device)

    x = F.normalize(x.float(), dim=-1)  # [B, D]
    sum_cos = x.sum(dim=0).square().sum() - B
    return sum_cos / (B * (B - 1))


@torch.no_grad()
def effective_rank(
    x: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Effective rank of feature covariance using entropy of eigenvalues.
    Higher = representation uses more dimensions.
    Lower = representation is concentrated in fewer directions.

    Args:
        x: feature tensor of shape [B, D]
        normalize: whether to L2-normalize features first
        eps: numerical stability

    Returns:
        scalar tensor in roughly [1, min(B - 1, D)]
    """

    x = x.float()
    B = x.shape[0]
    if B <= 1:
        return torch.tensor(1.0, device=x.device)

    if normalize:
        x = F.normalize(x, dim=-1)

    x = x - x.mean(dim=0, keepdim=True)  # [B, D]

    # Eigenvalues are proportional to singular values squared. Form the smaller Gram matrix to keep
    # patch-token diagnostics bounded by the projected embedding dimension instead of sample count.
    if B >= x.shape[1]:
        eigvals = torch.linalg.eigvalsh(x.T @ x)
    else:
        eigvals = torch.linalg.eigvalsh(x @ x.T)

    eigvals = eigvals[eigvals > eps]
    if eigvals.numel() == 0:
        return torch.tensor(1.0, device=x.device)

    p = eigvals / eigvals.sum()
    entropy = -(p * torch.log(p + eps)).sum()

    return torch.exp(entropy)
