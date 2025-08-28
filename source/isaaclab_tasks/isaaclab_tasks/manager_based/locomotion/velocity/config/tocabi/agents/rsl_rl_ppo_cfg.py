# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlPpoActorCriticSpectralNormCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg, 
    RslRlAMPAlgorithmCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlAMPConfig,
    RslRlDiscriminatorCfg,
    RslRlLcpCfg, 
    RslRlSymmetryCfg, 
    RslRlBoundLossCfg
)

@configclass
class TocabiRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 50
    experiment_name = "tocabi_rough"
    empirical_normalization = True
    # policy = RslRlPpoActorCriticRecurrentCfg(
        # rnn_type="lstm",
        # rnn_hidden_dim=256,
        # rnn_num_layers=1,
    # policy = RslRlPpoActorCriticSpectralNormCfg(
        # lipschitz_constant=0.2,
        # schedule="fixed",
        # schedule="learn",
        # lipschitz_coefficient=1.0,
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
        lcp_cfg=RslRlLcpCfg(
            gradient_penalty_coef=0.002,
            # is_lcp=False,
            is_lcp=True,
        ),
        bound_loss_cfg=RslRlBoundLossCfg(
            bound_loss_coef=10,
            bound_range=1.1,
        ),
    )

@configclass
class TocabiFlatPPORunnerCfg(TocabiRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "tocabi_flat_icra"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

@configclass
class TocabiMimicPPORunnerCfg(TocabiRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "tocabi_mimic"
        self.policy.actor_hidden_dims = [512, 512]
        self.policy.critic_hidden_dims = [512, 512]


@configclass
class TocabiAMPPPORunnerCfg(TocabiRoughPPORunnerCfg):
    algorithm = RslRlAMPAlgorithmCfg(
        value_loss_coef=5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
        lcp_cfg=RslRlLcpCfg(
            gradient_penalty_coef=0.005,
        ),
        bound_loss_cfg=RslRlBoundLossCfg(
            bound_loss_coef=10,
            bound_range=1.1,
        ),
        amp_cfg=RslRlAMPConfig(
            num_amp_obs_per_step=34,
            num_amp_obs_steps=2,
            disc_grad_pen=1.0,
            motion_file="source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/tocabi/motions/tocabi_motion.yaml",
            task_reward_ratio=0.5,
            disc_cfg=RslRlDiscriminatorCfg(
                num_amp_output=1,
                hidden_dims=[256, 256],
                activation="relu",
                gan_type="lsgan",
                amp_reward_weight=1.0,
            )
        )
    )

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "tocabi_amp"

@configclass
class TocabiDistillationStudentTeacherCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 500
    experiment_name = "tocabi_t2s"
    empirical_normalization = True
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-5,
        gradient_length=15,
        max_grad_norm=1.0,
    )

@configclass
class TocabiTeacherPPORunnerCfg(TocabiRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100000
        self.experiment_name = "tocabi_t2s"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
