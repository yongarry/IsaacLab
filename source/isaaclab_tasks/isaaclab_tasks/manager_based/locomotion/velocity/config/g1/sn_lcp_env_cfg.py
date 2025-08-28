# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG, G1_CFG  # isort: skip

@configclass
class G1_Actions:
    joint_pos = mdp.TocabiJointPositionActionCfg(
        asset_name="robot",
        lower_joint_names=["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                           "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
        upper_joint_names=["torso_joint",
                           "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_four_joint", "left_zero_joint", "left_one_joint", "left_two_joint",
                           "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_elbow_roll_joint", "right_five_joint", "right_three_joint", "right_four_joint", "right_zero_joint", "right_one_joint", "right_two_joint"],
        scale=0.5,
    )

@configclass
class G1_CMD:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )
    phase_time = mdp.WalkingPhaseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.WalkingPhaseCommandCfg.Ranges(
            phase_time=(1.0, 1.0)
        ),
    )

@configclass
class G1_Observations:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        clock_input = ObsTerm(func=mdp.clock_input, params={"command_name": "phase_time"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered_rel, 
                            noise=Unoise(n_min=-0.01, n_max=0.01),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                                                                                     "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered_rel, 
                            noise=Unoise(n_min=-1.5, n_max=1.5),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                                                                                     "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"])})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()

@configclass
class G1_Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0})
    root_bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.79, "asset_cfg": SceneEntityCfg("robot")})
    root_bad_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("robot")})

@configclass
class BipedalWalkingRewards:
    lin_vel_xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_base_frame_exp, weight=1.0, params={"command_name": "base_velocity", "omega": 8.0})
    ang_vel_z_tracking = RewTerm(func=mdp.track_ang_vel_z_world_exp_tocabi, weight=1.0, params={"command_name": "base_velocity", "omega": 7.0})
    orientation_tracking = RewTerm(func=mdp.orientation_tracking, weight=1.0, params={"omega": 5.0})
    base_height_tracking = RewTerm(func=mdp.base_height_tracking, weight=0.5, params={"height": 0.71, "omega": 10.0})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    periodic_force = RewTerm(func=mdp.periodic_force, weight=1.0, params={
                        # "scale": 500, 
                        "scale": 300, 
                        "command_name": "phase_time",
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                        "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"
    })
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    feet_height_tracking = RewTerm(func=mdp.feet_height_tracking, weight=1.0, params={
                        "omega": 100.0, "foot_height": 0.1, "kappa": 2.0, "offset": 0.0344,
                        # "omega": 5.0, "foot_height": 0.2, "kappa": 2.0, "offset": -0.77,
                        "command_name": "phase_time",
                        "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                        "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"})
    feet_velocity_z_tracking = RewTerm(func=mdp.feet_velocity_z_tracking, weight=0.5, params={
                        # "omega": 5.0, "foot_height": 0.1, "kappa": 4.0, "offset": 0.0344,
                        "omega": 80.0, "foot_height": 0.1, "kappa": 2.0,
                        "command_name": "phase_time",
                        "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                        "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    large_contact = RewTerm(func=mdp.large_contact_force, weight=-0.01, params={"threshold": 600.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                        "asset_cfg": SceneEntityCfg("robot")})
    default_joint = RewTerm(func=mdp.default_joint_pos, weight=0.5, params={"omega": 2.0, 
                        "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                                         "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"])})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipRoll_Joint"])})
    joint_deviation_ankle = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])})
    feet_flat = RewTerm(func=mdp.feet_flat, weight=0.1, params={"omega": 5.0, "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])})

@configclass
class G1_LCPEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: BipedalWalkingRewards = BipedalWalkingRewards()
    observations: G1_Observations = G1_Observations()
    commands: G1_CMD = G1_CMD()
    terminations: G1_Terminations = G1_Terminations()
    actions: G1_Actions = G1_Actions()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        # self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner =None
        self.curriculum.terrain_levels = None

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
        


@configclass
class G1_LCPEnvCfg_PLAY(G1_LCPEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
