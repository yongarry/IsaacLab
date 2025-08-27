# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import wrap_to_pi, quat_from_euler_xyz, quat_unique, quat_to_heading

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .cmd_cfgs import UniformFootCommandCfg


class UniformFootCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformFootCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformFootCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.episode_length_buf = env.episode_length_buf
        self.episode_time_s = env.episode_length_buf * env.step_dt
        self.lfoot_idx = self.robot.find_bodies(cfg.left_foot_name)[0][0]
        self.rfoot_idx = self.robot.find_bodies(cfg.right_foot_name)[0][0]
        
        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.is_lfoot_turn_to_resample = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pos_command_w = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        # self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)


    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass
        # # logs data
        # heading_to_quat = quat_from_euler_xyz(
        #     torch.zeros_like(self.heading_command_w),
        #     torch.zeros_like(self.heading_command_w),
        #     self.heading_command_w,
        # )
        # pos_error, rot_error = compute_pose_error(
        #     self.pos_command_w,
        #     heading_to_quat,
        #     self.robot.data.body_pos_w[:, self.body_idx],
        #     self.robot.data.body_quat_w[:, self.body_idx],
        # )
        # self.metrics["error_pos"] = torch.norm(pos_error, dim=-1)
        # self.metrics["error_rot"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        print(self.episode_length_buf[env_ids], self.episode_time_s)
        self.is_lfoot_turn_to_resample[env_ids] = torch.logical_not(self.is_lfoot_turn_to_resample[env_ids])

        lfoot_turn_to_resample = env_ids[self.is_lfoot_turn_to_resample[env_ids]]
        rfoot_turn_to_resample = env_ids[~self.is_lfoot_turn_to_resample[env_ids]]

        self.pos_command_w[lfoot_turn_to_resample, 0] = self.robot.data.body_pos_w[lfoot_turn_to_resample, self.lfoot_idx]
        self.pos_command_w[rfoot_turn_to_resample, 1] = self.robot.data.body_pos_w[rfoot_turn_to_resample, self.rfoot_idx]

        self.heading_command_w[lfoot_turn_to_resample, 0] = quat_to_heading(self.robot.data.body_quat_w[lfoot_turn_to_resample, self.lfoot_idx])
        self.heading_command_w[rfoot_turn_to_resample, 1] = quat_to_heading(self.robot.data.body_quat_w[rfoot_turn_to_resample, self.rfoot_idx])

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "lfoot_target_pose_visualizer"):
                # -- goal pose
                self.lfoot_target_pose_visualizer = VisualizationMarkers(self.cfg.lfoot_target_pose_visualizer_cfg)
                self.rfoot_target_pose_visualizer = VisualizationMarkers(self.cfg.rfoot_target_pose_visualizer_cfg)
            # set their visibility to true
            self.lfoot_target_pose_visualizer.set_visibility(True)
            self.rfoot_target_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "lfoot_target_pose_visualizer"):
                self.lfoot_target_pose_visualizer.set_visibility(False)
                self.rfoot_target_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.lfoot_target_pose_visualizer.visualize(
            translations=self.pos_command_w[:, 0],
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w[:, 0]),
                torch.zeros_like(self.heading_command_w[:, 0]),
                self.heading_command_w[:, 0],
            ),
        )
        self.rfoot_target_pose_visualizer.visualize(
            translations=self.pos_command_w[:, 1],
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w[:, 1]),
                torch.zeros_like(self.heading_command_w[:, 1]),
                self.heading_command_w[:, 1],
            ),
        )