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


def test_sigreg_init_weights_restores_buffers_after_nan_fill():
    loss_fn = SIGReg(n_knots=5, t_max=3.0, n_slices=7)
    loss_fn._apply(
        lambda t: torch.full_like(t, float("nan")) if t.dtype.is_floating_point else t,
        recurse=True,
    )

    loss_fn.init_weights()

    assert torch.isfinite(loss_fn.t).all()
    assert torch.isfinite(loss_fn.phi).all()
    assert torch.isfinite(loss_fn.weights).all()

    proj = torch.randn(3, 4, 8, requires_grad=True)
    loss = loss_fn(proj)

    assert torch.isfinite(loss)
    loss.backward()
    assert proj.grad is not None
    assert torch.isfinite(proj.grad).all()
