# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg
from .custom_cfg import RslRlLcpCfg, RslRlBoundLossCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log", "fixed"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

@configclass
class RslRlPpoActorCriticSpectralNormCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with spectral normalization."""

    class_name: str = "ActorCritic_SN"
    """The policy class name. Default is ActorCritic_SN."""
    
    lipschitz_constant: float = MISSING
    """The Lipschitz constant for the actor networks."""

    schedule: Literal["learn", "fixed"] = MISSING
    """The schedule for the Lipschitz constant. Either "learn" or "fixed"."""

    lipschitz_coefficient: float = 1.0
    """The coefficient for the Lipschitz constant."""

@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """

    lcp_cfg: RslRlLcpCfg | None = None
    """The configuration for the LCP module. Default is None, in which case LCP is not used."""

    bound_loss_cfg: RslRlBoundLossCfg | None = None
    """The configuration for the bound loss. Default is None, in which case bound loss is not used."""

@configclass
class RslRlAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP PPO algorithm."""

    class_name: str = "AMP"
    """The algorithm class name. Default is AMP."""

    amp_cfg: RslRlAMPConfig | None = None
    """The configuration for the AMP module. Default is None, in which case AMP is not used."""


@configclass
class RslRlAMPConfig:
    """Configuration for the AMP module."""

    num_amp_obs_per_step: int = MISSING
    """The number of AMP observations."""

    num_amp_obs_steps: int = 2
    """The number of AMP observations steps. Default is 2."""

    amp_replay_buffer_size: int = 100000
    """The size of the AMP replay buffer."""

    disc_loss_coef: float = 1.0
    """The loss coefficient for the discriminator."""
    
    disc_grad_pen: float = 1.0
    """The gradient penalty coefficient for the discriminator."""

    disc_logit_loss_coef: float = 0.05
    """The loss coefficient for the discriminator logit."""
    
    motion_file: str = MISSING
    """The path to the AMP motion file."""

    task_reward_ratio: float = 1.0
    """The ratio of the task reward to the AMP reward."""

    disc_cfg: RslRlDiscriminatorCfg | None = None
    """The configuration for the discriminator."""

@configclass
class RslRlDiscriminatorCfg:
    """Configuration for the discriminator."""
    
    num_amp_output: int = MISSING
    """The number of AMP output observations."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""

    activation: str = MISSING
    """The activation function for the discriminator network."""

    normalize_amp_obs: bool = True
    """Whether to normalize the AMP observations."""
    
    gan_type: Literal["wgan", "lsgan", "vanilla"] = "lsgan"
    """The type of GAN to use. Default is lsgan."""

    amp_reward_weight: float = 1.0
    """The weight for the AMP reward."""

#########################
# Runner configurations #
#########################


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg | RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillationAlgorithmCfg | RslRlAMPAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
