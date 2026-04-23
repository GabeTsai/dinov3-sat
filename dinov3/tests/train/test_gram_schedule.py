
import numpy as np
from omegaconf import OmegaConf

from dinov3.train.cosine_lr_scheduler import (
    build_gram_loss_weight_schedule,
    get_gram_loss_schedule_iteration,
    resolve_schedule_total_iterations,
)


def test_relative_gram_schedule_iteration_starts_at_first_update():
    assert (
        get_gram_loss_schedule_iteration(
            175_000,
            it_first_update=175_000,
            relative_to_first_update=True,
        )
        == 0
    )
    assert (
        get_gram_loss_schedule_iteration(
            175_123,
            it_first_update=175_000,
            relative_to_first_update=True,
        )
        == 123
    )


def test_relative_gram_schedule_iteration_clamps_before_first_update():
    assert (
        get_gram_loss_schedule_iteration(
            174_999,
            it_first_update=175_000,
            relative_to_first_update=True,
        )
        == 0
    )


def test_absolute_gram_schedule_iteration_preserves_existing_behavior():
    assert (
        get_gram_loss_schedule_iteration(
            175_000,
            it_first_update=175_000,
            relative_to_first_update=False,
        )
        == 175_000
    )


def test_relative_gram_schedule_length_and_warmup_are_stage_relative():
    schedule_cfg = OmegaConf.create(
        {
            "start": 0.0,
            "peak": 10.0,
            "end": 2.0,
            "warmup_epochs": 1,
            "cosine_epochs": 2,
        }
    )

    schedule = build_gram_loss_weight_schedule(
        schedule_cfg,
        iter_per_epoch=10,
        optim_epochs=30,
        it_first_update=100,
        relative_to_first_update=True,
    )

    assert len(schedule) == 200
    assert schedule[0] == 0.0
    assert schedule[10] == 10.0
    assert schedule[-1] == 2.0


def test_absolute_and_relative_schedules_choose_different_values_at_resume_iteration():
    schedule_cfg = OmegaConf.create(
        {
            "start": 0.0,
            "peak": 10.0,
            "end": 2.0,
            "warmup_epochs": 1,
            "cosine_epochs": 2,
        }
    )

    absolute_schedule = build_gram_loss_weight_schedule(
        schedule_cfg,
        iter_per_epoch=10,
        optim_epochs=30,
        it_first_update=100,
        relative_to_first_update=False,
    )
    relative_schedule = build_gram_loss_weight_schedule(
        schedule_cfg,
        iter_per_epoch=10,
        optim_epochs=30,
        it_first_update=100,
        relative_to_first_update=True,
    )

    absolute_index = get_gram_loss_schedule_iteration(
        100,
        it_first_update=100,
        relative_to_first_update=False,
    )
    relative_index = get_gram_loss_schedule_iteration(
        100,
        it_first_update=100,
        relative_to_first_update=True,
    )

    assert np.isclose(absolute_schedule[absolute_index], 2.0)
    assert relative_schedule[relative_index] == 0.0


def test_schedule_total_iterations_prefers_preserved_schedule_horizon():
    assert (
        resolve_schedule_total_iterations(
            iter_per_epoch=10,
            optim_epochs=120,
            schedule_epochs=100,
        )
        == 1000
    )


def test_relative_gram_schedule_can_preserve_original_total_iterations_when_extending_training():
    schedule_cfg = OmegaConf.create(
        {
            "start": 0.0,
            "peak": 10.0,
            "end": 2.0,
            "warmup_epochs": 1,
            "cosine_epochs": 2,
        }
    )

    schedule = build_gram_loss_weight_schedule(
        schedule_cfg,
        iter_per_epoch=10,
        optim_epochs=120,
        it_first_update=100,
        relative_to_first_update=True,
        schedule_total_iterations=1000,
    )

    assert len(schedule) == 900
    assert schedule[-1] == 2.0
