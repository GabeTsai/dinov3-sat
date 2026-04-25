from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dinov3.checkpointer import checkpointer
from dinov3.train import train


def test_trusted_legacy_load_planner_uses_weights_only_false(monkeypatch):
    planner = checkpointer.TrustedLegacyLoadPlanner(flatten_state_dict=False)
    planner.set_up_planner({})
    load_mock = Mock(return_value={"legacy": "value"})
    monkeypatch.setattr(checkpointer.torch, "load", load_mock)

    planner.load_bytes(
        SimpleNamespace(dest_index=SimpleNamespace(fqn="iteration")),
        BytesIO(b"legacy"),
    )

    load_mock.assert_called_once()
    _, kwargs = load_mock.call_args
    assert kwargs["weights_only"] is False
    assert planner.state_dict["iteration"] == {"legacy": "value"}


@pytest.mark.parametrize(
    ("trusted_legacy_bytes", "expected_planner_type"),
    [
        (False, checkpointer.dcp.default_planner.DefaultLoadPlanner),
        (True, checkpointer.TrustedLegacyLoadPlanner),
    ],
)
def test_load_checkpoint_selects_expected_planner(monkeypatch, trusted_legacy_bytes, expected_planner_type):
    captured = {}

    monkeypatch.setattr(checkpointer.dcpsd, "get_model_state_dict", lambda model: {})
    monkeypatch.setattr(checkpointer.dcpsd, "set_model_state_dict", lambda model, state_dict: None)
    monkeypatch.setattr(checkpointer.dcpfs, "FileSystemReader", lambda ckpt_dir: ("reader", ckpt_dir))
    monkeypatch.setattr(checkpointer.dist, "is_initialized", lambda: False)

    def fake_dcp_load(state_dict, *, storage_reader, planner, process_group):
        captured["storage_reader"] = storage_reader
        captured["planner"] = planner
        state_dict["iteration"] = 7

    monkeypatch.setattr(checkpointer.dcp, "load", fake_dcp_load)

    iteration = checkpointer.load_checkpoint(
        "trusted/legacy/ckpt",
        model=object(),
        trusted_legacy_bytes=trusted_legacy_bytes,
    )

    assert iteration == 7
    assert captured["storage_reader"] == ("reader", Path("trusted/legacy/ckpt"))
    assert isinstance(captured["planner"], expected_planner_type)


def test_get_args_parser_parses_trusted_legacy_resume_flag():
    args = train.get_args_parser().parse_args(["--trusted-legacy-resume"])

    assert args.trusted_legacy_resume is True


def test_do_train_passes_trusted_legacy_resume_to_load_checkpoint(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        train=SimpleNamespace(output_dir=str(tmp_path), OFFICIAL_EPOCH_LENGTH=1, batch_size_per_gpu=1),
        multidistillation=SimpleNamespace(enabled=False),
        optim=SimpleNamespace(epochs=0),
        checkpointing=SimpleNamespace(period=1, max_to_keep=1),
        gram=SimpleNamespace(use_loss=False),
        evaluation=SimpleNamespace(eval_period_iterations=0),
    )
    model = SimpleNamespace(
        train=lambda: None,
        get_params_groups=lambda: [],
        init_weights=lambda resume_ckpt_dir=None: None,
        student=object(),
    )
    load_kwargs = {}

    monkeypatch.setattr(train.distributed, "get_process_subgroup", lambda: None)
    monkeypatch.setattr(train.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(train.distributed, "is_main_process", lambda: False)
    monkeypatch.setattr(train.distributed, "is_subgroup_main_process", lambda: False)
    monkeypatch.setattr(train, "build_optimizer", lambda cfg, params_groups: object())
    monkeypatch.setattr(train, "build_schedulers", lambda cfg: ([0], [0], [0], [0], [0]))
    monkeypatch.setattr(train, "find_latest_checkpoint", lambda ckpt_dir: tmp_path / "ckpt" / "10")
    monkeypatch.setattr(train, "build_multi_resolution_data_loader_from_cfg", lambda cfg, model, start_iter: [])
    monkeypatch.setattr(train, "get_schedule_state", lambda cfg: {"schedule_total_iterations": 0})
    monkeypatch.setattr(train, "get_num_gram_updates_before_start", lambda cfg, model, start_iter: 0)

    def fake_load_checkpoint(*args, **kwargs):
        load_kwargs.update(kwargs)
        return -1

    monkeypatch.setattr(train, "load_checkpoint", fake_load_checkpoint)

    train.do_train(cfg, model, resume=True, trusted_legacy_resume=True)

    assert load_kwargs["trusted_legacy_bytes"] is True


def test_apply_optim_scheduler_casts_numpy_scalars_to_python_floats():
    optimizer = SimpleNamespace(
        param_groups=[
            {
                "is_last_layer": False,
                "lr_multiplier": np.float64(0.5),
                "wd_multiplier": np.float64(2.0),
            },
            {
                "is_last_layer": True,
                "lr_multiplier": np.float64(0.25),
                "wd_multiplier": np.float64(0.0),
            },
        ]
    )

    train.apply_optim_scheduler(
        optimizer,
        lr=np.float64(0.1),
        wd=np.float64(0.2),
        last_layer_lr=np.float64(0.3),
    )

    assert optimizer.param_groups[0]["lr"] == 0.05
    assert type(optimizer.param_groups[0]["lr"]) is float
    assert optimizer.param_groups[0]["weight_decay"] == 0.4
    assert type(optimizer.param_groups[0]["weight_decay"]) is float

    assert optimizer.param_groups[1]["lr"] == 0.075
    assert type(optimizer.param_groups[1]["lr"]) is float
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert type(optimizer.param_groups[1]["weight_decay"]) is float
