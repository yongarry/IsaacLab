# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_torques_l2_exp(env: ManagerBasedRLEnv, alpha: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.exp(-alpha*torch.norm(asset.data.applied_torque[:, asset_cfg.joint_ids], dim=1))


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def in_the_air(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalizes the robot for lifting both feet off the ground.

    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    reward = torch.where(torch.sum(contacts, dim=1) > 0, 1.0, 0.0)
    return reward

def feet_contact_force(env, threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """This reward penalizes overly high contact force on the robot’s feet
    
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    threshold_ = threshold * 9.81 * asset.data.default_mass[:, asset_cfg.body_ids].sum(dim=1).unsqueeze(1).repeat(1, len(sensor_cfg.body_ids)).to(contact_force.device)
    reward = torch.sum(torch.where(contact_force > threshold_, contact_force - threshold_, 0.0), dim=1)
    return reward

def feet_force(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] < 1.0

    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    reward = torch.sum(contact_force * contacts, dim=1)
    return reward

def joint_deviation_neg(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate for negative values on tocabi knee joint"""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] 
    # return torch.sum(torch.abs(angle), dim=1)
    return torch.sum(torch.where(angle > 0, 0.0, angle), dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_xy_base_frame_exp(
    env, omega: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the base frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-omega*lin_vel_error)

def track_ang_vel_z_world_exp_tocabi(
    env, command_name: str, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-omega*ang_vel_error)

def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def contact_force(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """this is only for plotting when playing"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return contact_force.squeeze()


from isaaclab_tasks.manager_based.locomotion.velocity.config.tocabi.motions.motion_loader import MotionLoader
# -----------------------------------------------------------------------------
# Deep Mimic motion cache (avoid reloading file each simulation step)
# -----------------------------------------------------------------------------
_DEEP_MIMIC_MOTION_CACHE: dict[str, MotionLoader] = {}

def _get_motion_loader(motion_file: str, device: torch.device | str):
    """Return cached MotionLoader, loading if necessary."""
    if isinstance(device, str):
        device = torch.device(device)
    if motion_file not in _DEEP_MIMIC_MOTION_CACHE:
        _DEEP_MIMIC_MOTION_CACHE[motion_file] = MotionLoader(motion_file, device=device)
    return _DEEP_MIMIC_MOTION_CACHE[motion_file]


def deep_mimic_rewards(
    env,
    step_time: float,
    alpha: float,
    motion_file: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of whole-body motion from motion capture data.

    Args:
        env: RL 환경.
        step_time: 모션이 한 주기를 도는 시간(s).
        motion_file: NPZ 또는 TXT 등 모션 파일 경로 (문자열).
        asset_cfg: 로봇 SceneEntity 설정.
    """
    # Lazy-load & cache motion
    motion = _get_motion_loader(motion_file, device=env.device)

    # 현재 phase 계산 (0~duration)
    # 5600 ~ 9201 is the range of the motion data (2.8s ~ 4.6s)
    phase = (env.episode_length_buf.unsqueeze(1) * env.step_dt % step_time) + 2.8
    joint_position, _, _, _, _, _, _ = motion.sample(1, times=phase.cpu().numpy().reshape(-1))
    # rearange the joint position to the order of the asset_cfg
    asset = env.scene[asset_cfg.name]
    # get the reference joint order in motion that matches the asset's joint ids
    joint_names_selected = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
    selected_joints = motion.get_dof_index(joint_names_selected)

    # reference and current joint positions
    joint_position_ref = joint_position[:, selected_joints]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # compute reward with exponential kernel
    reward = torch.exp(-alpha * torch.norm(joint_position_ref - current_joint_pos, dim=1) ** 2)
    return reward


# -----------------------------------------------------------------------------
# World Model Rewards
# -----------------------------------------------------------------------------
def orientation_tracking(env, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ maintain base orientation(alpha, beta) flat in base frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.exp(-omega*(torch.square(roll) + torch.square(pitch)))

def feet_flat(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ give penalty if the projected gravity vector is not flat"""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    for ids in asset_cfg.body_ids:
        link_quat_w = asset.data.body_link_quat_w[:, ids, :].squeeze(1)
        gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
        projected_gravity = math_utils.quat_apply_inverse(link_quat_w, gravity)
        reward += torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return reward

def base_height_tracking(env, height: float, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ maintain base height(z) flat in base frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.exp(-omega*torch.square(asset.data.root_pos_w[:, 2] - height))

def periodic_force(
        env, 
        scale: float, 
        command_name: str, 
        left_foot: str, 
        right_foot: str, 
        is_rfoot_first: str,
        sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward periodic force on the robot's feet. reasonable foot forces during the stance phase of locomotion."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    phase_time = env.command_manager.get_command(command_name)
    is_rfoot = env.command_manager.get_command(is_rfoot_first)

    cycle_time = env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time
    second_foot_indicator = torch.where(cycle_time < phase_time/2, 1.0, 0.0) # left foot
    first_foot_indicator = torch.where(cycle_time >= phase_time/2, 1.0, 0.0) # right foot

    left_foot_indicator = torch.where(~is_rfoot, first_foot_indicator, second_foot_indicator)
    right_foot_indicator = torch.where(is_rfoot, first_foot_indicator, second_foot_indicator)

    left_foot_idx = [sensor_cfg.body_ids[i] for i in range(len(sensor_cfg.body_ids)) if left_foot in sensor_cfg.body_names[i]]
    right_foot_idx = [sensor_cfg.body_ids[i] for i in range(len(sensor_cfg.body_ids)) if right_foot in sensor_cfg.body_names[i]]

    left_foot_force = contact_sensor.data.net_forces_w_history[:, :, left_foot_idx, :].norm(dim=-1).max(dim=1)[0]
    right_foot_force = contact_sensor.data.net_forces_w_history[:, :, right_foot_idx, :].norm(dim=-1).max(dim=1)[0]
    reward = left_foot_indicator * left_foot_force / scale + right_foot_indicator * right_foot_force / scale
    reward = torch.clip(reward, 0.0, 1.0)
    return reward.squeeze()

def periodic_velocity(
        env, 
        scale: float, 
        command_name: str, 
        is_rfoot_first: str,
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward periodic velocity on the robot's feet. reasonable foot velocity during the stance phase of locomotion."""
    asset = env.scene[asset_cfg.name]
    phase_time = env.command_manager.get_command(command_name)
    is_rfoot = env.command_manager.get_command(is_rfoot_first)

    cycle_time = env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time
    second_foot_indicator = torch.where(cycle_time < phase_time/2, 1.0, 0.0) # left foot
    first_foot_indicator = torch.where(cycle_time >= phase_time/2, 1.0, 0.0) # right foot

    left_foot_indicator = torch.where(~is_rfoot, first_foot_indicator, second_foot_indicator)
    right_foot_indicator = torch.where(is_rfoot, first_foot_indicator, second_foot_indicator)

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    left_foot_vel = torch.norm(asset.data.body_lin_vel_w[:, left_foot_idx, :], dim=-1)
    right_foot_vel = torch.norm(asset.data.body_lin_vel_w[:, right_foot_idx, :], dim=-1)

    reward = (1 - left_foot_indicator) * left_foot_vel * scale + (1 - right_foot_indicator) * right_foot_vel * scale
    reward = torch.clip(reward, 0.0, 1.0)
    return reward.squeeze()

def feet_height_tracking(
        env, 
        omega: float, 
        foot_height: float, 
        kappa: float, 
        offset: float, 
        command_name: str, 
        is_rfoot_first: str,
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet height trajectory. generated by Von Mises distribution."""
    asset = env.scene[asset_cfg.name]
    phase_time = env.command_manager.get_command(command_name)
    is_rfoot = env.command_manager.get_command(is_rfoot_first)

    B = torch.exp(-torch.tensor(kappa))
    norm = torch.exp(torch.tensor(kappa))-torch.exp(-torch.tensor(kappa))
    theta1 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time) / phase_time - math.pi/2 #right foot
    theta2 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt - phase_time/2) % phase_time / phase_time - math.pi/2 #left foot
    gx1 = torch.exp(torch.tensor(kappa) * torch.cos(theta1))
    gx2 = torch.exp(torch.tensor(kappa) * torch.cos(theta2))
    first_foot_traj = foot_height * (gx1-B) / norm + offset
    second_foot_traj = foot_height * (gx2-B) / norm + offset

    left_foot_traj = torch.where(~is_rfoot, first_foot_traj, second_foot_traj)
    right_foot_traj = torch.where(is_rfoot, first_foot_traj, second_foot_traj)

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    # change to tracking based on base frame 
    l_foot_pos_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, left_foot_idx[0], :3] - asset.data.root_pos_w[:, :3])
    r_foot_pos_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, right_foot_idx[0], :3] - asset.data.root_pos_w[:, :3])

    # left_foot_pos_z = asset.data.body_pos_w[:, left_foot_idx, 2] - left_foot_traj
    # right_foot_pos_z = asset.data.body_pos_w[:, right_foot_idx, 2] - right_foot_traj
    left_foot_pos_z = l_foot_pos_b[:, 2].unsqueeze(1) - left_foot_traj
    right_foot_pos_z = r_foot_pos_b[:, 2].unsqueeze(1) - right_foot_traj
    foot_tracking_error = torch.sum(torch.square(left_foot_pos_z) + torch.square(right_foot_pos_z), dim=1).squeeze()
    return torch.exp(-omega*foot_tracking_error)

def feet_velocity_z_tracking(
        env, 
        omega: float, 
        foot_height: float, 
        kappa: float, 
        command_name: str, 
        is_rfoot_first: str,
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet velocity trajectory. generated by differentiated Von Mises distribution."""
    asset = env.scene[asset_cfg.name]
    phase_time = env.command_manager.get_command(command_name)
    is_rfoot = env.command_manager.get_command(is_rfoot_first)

    B = torch.exp(-torch.tensor(kappa))
    norm = torch.exp(torch.tensor(kappa))-torch.exp(-torch.tensor(kappa))
    theta_right = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time) / phase_time - math.pi/2
    theta_left = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt - phase_time/2) % phase_time / phase_time - math.pi/2
    gx_right = torch.exp(torch.tensor(kappa) * torch.cos(theta_right))
    gx_left = torch.exp(torch.tensor(kappa) * torch.cos(theta_left))
    dtheta = 2 * math.pi / phase_time
    d_gx_right = -kappa * torch.sin(theta_right) * gx_right * dtheta
    d_gx_left = -kappa * torch.sin(theta_left) * gx_left * dtheta

    foot_traj_first = foot_height * (d_gx_right) / norm #right foot
    foot_traj_second = foot_height * (d_gx_left) / norm #left foot

    foot_traj_left = torch.where(~is_rfoot, foot_traj_first, foot_traj_second)
    foot_traj_right = torch.where(is_rfoot, foot_traj_first, foot_traj_second)

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    left_foot_vel_z_error = asset.data.body_lin_vel_w[:, left_foot_idx, 2] - foot_traj_left
    right_foot_vel_z_error = asset.data.body_lin_vel_w[:, right_foot_idx, 2] - foot_traj_right

    # left_foot_vel_y_error = asset.data.body_lin_vel_w[:, left_foot_idx, 1]
    # right_foot_vel_y_error = asset.data.body_lin_vel_w[:, right_foot_idx, 1]

    foot_vel_tracking_error = torch.sum(torch.square(left_foot_vel_z_error) + torch.square(right_foot_vel_z_error), dim=1).squeeze()
    return torch.exp(-omega*foot_vel_tracking_error)

def feet_height_poly_tracking(env, omega: float, foot_height: float, phase_time: float, kappa: float, offset: float, left_foot: str, right_foot: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet height trajectory. generated by polynomial function."""
    asset = env.scene[asset_cfg.name]

    B = torch.exp(-torch.tensor(kappa))
    norm = torch.exp(torch.tensor(kappa))-torch.exp(-torch.tensor(kappa))
    theta1 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time) / phase_time - math.pi/2 #right foot
    theta2 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt - phase_time/2) % phase_time / phase_time - math.pi/2 #left foot
    gx1 = torch.exp(torch.tensor(kappa) * torch.cos(theta1))
    gx2 = torch.exp(torch.tensor(kappa) * torch.cos(theta2))
    right_foot_traj = foot_height * (gx1-B) / norm + offset
    left_foot_traj = foot_height * (gx2-B) / norm + offset


    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    l_foot_pos_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, left_foot_idx[0], :] - asset.data.root_pos_w[:, :3])
    r_foot_pos_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, right_foot_idx[0], :] - asset.data.root_pos_w[:, :3])

    # left_foot_pos_z = asset.data.body_pos_w[:, left_foot_idx, 2] - left_foot_traj
    # right_foot_pos_z = asset.data.body_pos_w[:, right_foot_idx, 2] - right_foot_traj
    left_foot_pos_z = l_foot_pos_b[:, 2].unsqueeze(1) - left_foot_traj
    right_foot_pos_z = r_foot_pos_b[:, 2].unsqueeze(1) - right_foot_traj

    foot_tracking_error = torch.sum(torch.square(left_foot_pos_z) + torch.square(right_foot_pos_z), dim=1).squeeze()
    return torch.exp(-omega*foot_tracking_error)

def large_contact_force(env, threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize large contact force on the robot's feet."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    max_force = 1.3 * 9.81 * asset.data.default_mass[:, asset_cfg.body_ids].sum(dim=1).unsqueeze(1).to(contact_forces.device)
    clipped_contact_forces = torch.clip(contact_forces-max_force, 0.0, threshold)
    reward = torch.sum(clipped_contact_forces, dim=1)
    return reward

def default_joint_pos(env, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward default joint positions."""
    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.norm(default_joint_pos - current_joint_pos, dim=1).squeeze()
    return torch.exp(-omega*error)

def feet_velocity_y_tracking(env, omega: float, left_foot: str, right_foot: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet velocity trajectory. generated by differentiated Von Mises distribution."""
    asset = env.scene[asset_cfg.name]

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    # get the velocity of the feet in the y direction on the base frame
    left_foot_vel_y_error = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, asset.data.body_lin_vel_w[:, left_foot_idx, :])[..., 1]
    right_foot_vel_y_error = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, asset.data.body_lin_vel_w[:, right_foot_idx, :])[..., 1]

    foot_vel_tracking_error = torch.sum(torch.square(left_foot_vel_y_error) + torch.square(right_foot_vel_y_error), dim=1).squeeze()
    return torch.exp(-omega*foot_vel_tracking_error)

# def feet_x_tracking(env, omega: float, phase_time: float, command_name: str, left_foot: str, right_foot: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Reward tracking of feet x position trajectory. generated by polynomial function."""
#     asset = env.scene[asset_cfg.name]
#     command_vel_x = env.command_manager.get_command(command_name)[:, 0]

#     left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
#     right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

#     left_foot_pos_x = asset.data.body_pos_w[:, left_foot_idx, 0]
#     right_foot_pos_x = asset.data.body_pos_w[:, right_foot_idx, 0]