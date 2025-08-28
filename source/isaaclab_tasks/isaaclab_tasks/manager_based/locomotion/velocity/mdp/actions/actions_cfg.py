from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import joint_actions_tocabi

##
# Joint actions.
##


@configclass
class TocabiActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = joint_actions_tocabi.TocabiAction

    lower_joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    
    upper_joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    rescale_to_limits: bool = True
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """

    pd_control: bool = True
    """Whether to use PD control. Defaults to True."""

    joint_pos_limits: list[tuple[float, float]] = MISSING

    torque_limits: list[float] = MISSING
    """Torque limits for the action (float or dict of regex expressions). Only used when pd_control is False."""

    p_gains: list[float] = MISSING
    """P-gain for the action (float or dict of regex expressions)."""
    
    d_gains: list[float] = MISSING
    """D-gain for the action (float or dict of regex expressions)."""

    rand_torque_inj_range: tuple[float, float] = (0.0, 0.0)
    """Range for the random torque injection. Defaults to (0.0, 0.0)."""

    rand_motor_scale_range: tuple[float, float] = (1.0, 1.0)
    """Range for the random motor scale. Defaults to (0.8, 1.2)."""

@configclass
class TocabiJointPositionActionCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions_tocabi.JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    lower_joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    
    upper_joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""