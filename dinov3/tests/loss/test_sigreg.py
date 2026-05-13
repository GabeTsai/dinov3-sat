import torch

from dinov3.loss import SIGReg


def test_sigreg_returns_scalar_and_backpropagates():
    proj = torch.randn(3, 4, 8, requires_grad=True)
    loss = SIGReg(n_knots=5, t_max=3.0, n_slices=7)(proj)

    assert loss.shape == ()
    assert torch.isfinite(loss)

    loss.backward()
    assert proj.grad is not None
    assert torch.isfinite(proj.grad).all()
