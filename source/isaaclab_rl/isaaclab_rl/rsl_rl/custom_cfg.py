# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlLcpCfg:
    """Configuration for the LCP module."""
    gradient_penalty_coef: float = 0.0
    """The coefficient for the gradient penalty loss."""
    is_lcp: bool = True
    """Whether to use the LCP module."""

    gradient_penalty_coef_schedule: list[float] = MISSING
    """The schedule for the gradient penalty coefficient."""

    gradient_penalty_coef_schedule_steps: list[int] = MISSING
    """The steps for the gradient penalty coefficient schedule."""
    

@configclass
class RslRlBoundLossCfg:
    """Configuration for the bound loss."""
    bound_loss_coef: float = 10.0
    """The coefficient for the bound loss."""

    bound_range: float = 1.1