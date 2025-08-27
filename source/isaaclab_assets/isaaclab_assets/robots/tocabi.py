# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`Tocabi_CFG`: Tocabi humanoid robot
* :obj:`Tocabi_MINIMAL_CFG`: Tocabi humanoid robot with minimal collision bodies
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

##
# Configuration - Actuators.
##

Tocabi_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{os.getcwd()}/source/isaaclab_assets/data/Robots/Tocabi/tocabi.usd",
        usd_path=f"{os.getcwd()}/source/isaaclab_assets/data/Robots/Tocabi/dyros_tocabi.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions= True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            ".*_HipYaw_Joint": 0.0,
            ".*_HipRoll_Joint": 0.0,
            # ".*_HipPitch_Joint": -0.5, 
            ".*_HipPitch_Joint": -0.24,
            # ".*_Knee_Joint": 1.0,
            ".*_Knee_Joint": 0.6,
            # ".*_AnklePitch_Joint": -0.5,
            ".*_AnklePitch_Joint": -0.36,
            ".*_AnkleRoll_Joint": 0.0,
            "Waist1_Joint": 0.0,
            "Waist2_Joint": 0.0,
            "Upperbody_Joint": 0.0,
            "L_Shoulder1_Joint": 0.3,
            "L_Shoulder2_Joint": 0.3,
            "L_Shoulder3_Joint": 1.5,
            "L_Armlink_Joint": -1.27,
            "L_Elbow_Joint": -1.0,
            "L_Forearm_Joint": 0.0,
            "L_Wrist1_Joint": -1.0,
            "L_Wrist2_Joint": 0.0,
            "Neck_Joint": 0.0,
            "Head_Joint": 0.0,
            "R_Shoulder1_Joint": -0.3,
            "R_Shoulder2_Joint": -0.3,
            "R_Shoulder3_Joint": -1.5,
            "R_Armlink_Joint": 1.27,
            "R_Elbow_Joint": 1.0,
            "R_Forearm_Joint": 0.0,
            "R_Wrist1_Joint": 1.0,
            "R_Wrist2_Joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    # actuators={
    #     "lowerbody": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             ".*_HipYaw_Joint",
    #             ".*_HipRoll_Joint",
    #             ".*_HipPitch_Joint",
    #             ".*_Knee_Joint",
    #             ".*_AnklePitch_Joint",
    #             ".*_AnkleRoll_Joint"],
    #         stiffness={".*": 0.0},
    #         damping={".*": 1.0},
    #         effort_limit_sim={
    #             ".*_HipYaw_Joint": 333,
    #             ".*_HipRoll_Joint": 232,
    #             ".*_HipPitch_Joint": 263,
    #             ".*_Knee_Joint": 289,
    #             ".*_AnklePitch_Joint": 222,
    #             ".*_AnkleRoll_Joint": 166,
    #         },
    #         velocity_limit_sim=4.03,
    #         armature={
    #             ".*_HipYaw_Joint": 0.614,
    #             ".*_HipRoll_Joint": 0.862,
    #             ".*_HipPitch_Joint": 1.09,
    #             ".*_Knee_Joint": 1.09,
    #             ".*_AnklePitch_Joint": 1.09,
    #             ".*_AnkleRoll_Joint": 0.360,
    #         },
    #     ),
    #     "upperbody": ImplicitActuatorCfg(
    #         joint_names_expr=[
    #             "Waist1_Joint",
    #             "Waist2_Joint",
    #             "Upperbody_Joint",
    #             ".*_Shoulder1_Joint",
    #             ".*_Shoulder2_Joint",
    #             ".*_Shoulder3_Joint",
    #             ".*_Armlink_Joint",
    #             ".*_Elbow_Joint",
    #             ".*_Forearm_Joint",
    #             ".*_Wrist1_Joint",
    #             ".*_Wrist2_Joint",
    #             "Neck_Joint",
    #             "Head_Joint"
    #         ],
    #         stiffness={".*": 0.0},
    #         damping={".*": 1.0},
    #         effort_limit_sim={
    #             "Waist1_Joint": 303,
    #             "Waist2_Joint": 303,
    #             "Upperbody_Joint": 303,
    #             ".*_Shoulder1_Joint": 64,
    #             ".*_Shoulder2_Joint": 64,
    #             ".*_Shoulder3_Joint": 64,
    #             ".*_Armlink_Joint": 64,
    #             ".*_Elbow_Joint": 23,
    #             ".*_Forearm_Joint": 23,
    #             ".*_Wrist1_Joint": 10,
    #             ".*_Wrist2_Joint": 10,
    #             "Neck_Joint": 10,
    #             "Head_Joint": 10
    #         },
    #         velocity_limit_sim=4.03,
    #         armature={
    #             "Waist1_Joint": 0.078,
    #             "Waist2_Joint": 0.078,
    #             "Upperbody_Joint": 0.078,
    #             ".*_Shoulder1_Joint": 0.18,
    #             ".*_Shoulder2_Joint": 0.18,
    #             ".*_Shoulder3_Joint": 0.18,
    #             ".*_Armlink_Joint": 0.18,
    #             ".*_Elbow_Joint": 0.0032,
    #             ".*_Forearm_Joint": 0.0032,
    #             ".*_Wrist1_Joint": 0.0032,
    #             ".*_Wrist2_Joint": 0.0032,
    #             "Neck_Joint": 0.0032,
    #             "Head_Joint": 0.0032},
    #     ),
    # },
    actuators={
        "tocabi": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_HipYaw_Joint",
                ".*_HipRoll_Joint",
                ".*_HipPitch_Joint",
                ".*_Knee_Joint",
                ".*_AnklePitch_Joint",
                ".*_AnkleRoll_Joint",
                "Waist1_Joint",
                "Waist2_Joint",
                "Upperbody_Joint",
                ".*_Shoulder1_Joint",
                ".*_Shoulder2_Joint",
                ".*_Shoulder3_Joint",
                ".*_Armlink_Joint",
                ".*_Elbow_Joint",
                ".*_Forearm_Joint",
                ".*_Wrist1_Joint",
                ".*_Wrist2_Joint",
                "Neck_Joint",
                "Head_Joint"
            ],
            stiffness={".*": 0.0},
            damping={".*": 1.0},
            effort_limit_sim={
                ".*_HipYaw_Joint": 333,
                ".*_HipRoll_Joint": 232,
                ".*_HipPitch_Joint": 263,
                ".*_Knee_Joint": 289,
                ".*_AnklePitch_Joint": 222,
                ".*_AnkleRoll_Joint": 166,
                "Waist1_Joint": 303,
                "Waist2_Joint": 303,
                "Upperbody_Joint": 303,
                ".*_Shoulder1_Joint": 64,
                ".*_Shoulder2_Joint": 64,
                ".*_Shoulder3_Joint": 64,
                ".*_Armlink_Joint": 64,
                ".*_Elbow_Joint": 23,
                ".*_Forearm_Joint": 23,
                ".*_Wrist1_Joint": 10,
                ".*_Wrist2_Joint": 10,
                "Neck_Joint": 10,
                "Head_Joint": 10
            },
            velocity_limit_sim=4.03,
            armature={
                ".*_HipYaw_Joint": 0.614,
                ".*_HipRoll_Joint": 0.862,
                ".*_HipPitch_Joint": 1.09,
                ".*_Knee_Joint": 1.09,
                ".*_AnklePitch_Joint": 1.09,
                ".*_AnkleRoll_Joint": 0.360,
                "Waist1_Joint": 0.078,
                "Waist2_Joint": 0.078,
                "Upperbody_Joint": 0.078,
                ".*_Shoulder1_Joint": 0.18,
                ".*_Shoulder2_Joint": 0.18,
                ".*_Shoulder3_Joint": 0.18,
                ".*_Armlink_Joint": 0.18,
                ".*_Elbow_Joint": 0.0032,
                ".*_Forearm_Joint": 0.0032,
                ".*_Wrist1_Joint": 0.0032,
                ".*_Wrist2_Joint": 0.0032,
                "Neck_Joint": 0.0032,
                "Head_Joint": 0.0032
            },
        ),
    },
)
"""Configuration for the Tocabi Humanoid robot."""


# Tocabi_MINIMAL_CFG = Tocabi_CFG.copy()
# Tocabi_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Tocabi/Tocabi_minimal.usd"
"""Configuration for the Tocabi Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""


