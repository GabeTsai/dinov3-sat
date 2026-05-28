from omegaconf import OmegaConf

from dinov3.train import train


def test_scalar_keep_every_preserves_periodic_save_behavior():
    cfg = OmegaConf.create({"checkpointing": {"period": 5, "keep_every": 10}})

    assert not train.should_save_checkpoint(cfg, 8)
    assert train.should_save_checkpoint(cfg, 9)
    assert train.should_keep_checkpoint_copy(cfg.checkpointing.keep_every, 9)


def test_list_keep_every_saves_and_keeps_exact_iterations():
    cfg = OmegaConf.create({"checkpointing": {"period": 5, "keep_every": [3, 11]}})

    assert train.should_save_checkpoint(cfg, 3)
    assert train.should_keep_checkpoint_copy(cfg.checkpointing.keep_every, 3)
    assert train.should_save_checkpoint(cfg, 11)
    assert train.should_keep_checkpoint_copy(cfg.checkpointing.keep_every, 11)


def test_list_keep_every_does_not_keep_periodic_checkpoints_not_in_list():
    cfg = OmegaConf.create({"checkpointing": {"period": 5, "keep_every": [3, 11]}})

    assert train.should_save_checkpoint(cfg, 4)
    assert not train.should_keep_checkpoint_copy(cfg.checkpointing.keep_every, 4)
