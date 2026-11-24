"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.build.multitask import (
    LikGaussTask,
    TaskCov,
    TaskMean,
    augment,
    task_expand,
    task_index,
)

__all__ = ["LikGaussTask", "TaskCov", "TaskMean", "augment", "task_expand", "task_index"]
