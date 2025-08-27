import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.utils import configclass

from .phase_time_cmd import WalkingPhaseCommand, FirstFootStepCommand
from .foot_command import UniformFootCommand

@configclass
class WalkingPhaseCommandCfg(CommandTermCfg):
    """Configuration for the walking phase command generator."""

    class_type: type = WalkingPhaseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the walking phase commands."""

        phase_time: tuple[float, float] = MISSING
        """Range for the phase time (in s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the walking phase commands."""

@configclass
class FirstFootStepCommandCfg(CommandTermCfg):
    """Configuration for the first foot step command generator."""

    class_type: type = FirstFootStepCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    is_rfoot_first: float = 0.5
    """Probability of the right foot being the first foot."""

@configclass
class UniformFootCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformFootCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_foot_name: str = "L_AnkleRoll_Link"
    right_foot_name: str = "R_AnkleRoll_Link"

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    left_ranges: Ranges = MISSING
    right_ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    lfoot_target_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/lfoot_pose_goal"
    )
    lfoot_target_pose_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.3)

    rfoot_target_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/rfoot_pose_goal"
    )
    rfoot_target_pose_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.3)