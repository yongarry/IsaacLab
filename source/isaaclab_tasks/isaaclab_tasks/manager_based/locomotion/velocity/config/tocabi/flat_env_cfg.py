# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from .rough_env_cfg import TocabiRoughEnvCfg, TocabiRewards
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

@configclass
class TocabiFlatEnvCfg(TocabiRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

class TocabiFlatEnvCfg_PLAY(TocabiFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.episode_length_s = 10.0
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.viewer.resolution = (2560, 1440)
        # self.viewer.resolution = (1920, 1080)
        self.viewer.eye = (-3.0, 3.0, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = -1

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.push_robot = None
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        self.rewards.contact_force_l = RewTerm(
            func=mdp.contact_force,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="L_AnkleRoll.*")},
        )
        self.rewards.contact_force_r = RewTerm(
            func=mdp.contact_force,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="R_AnkleRoll.*")},
        )




@configclass
class TocabiMimicRewards(TocabiRewards):
    deep_mimic_reward = RewTerm(
        func=mdp.deep_mimic_rewards,
        weight=1.0,
        params={
            "alpha": 2.0,
            "step_time": 1.8,
            "asset_cfg": SceneEntityCfg("robot", 
                                        joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"]),
            "motion_file": "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/tocabi/motions/tocabi_0.9walk.txt"
        },
    )
@configclass
class TocabiMimicEnvCfg(TocabiFlatEnvCfg):
    rewards: TocabiMimicRewards = TocabiMimicRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.feet_air_time = None
        self.rewards.feet_slide = None
        self.rewards.feet_contact_force = None

class TocabiMimicEnvCfg_PLAY(TocabiMimicEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None