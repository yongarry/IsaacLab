import isaaclab.sim as sim_utils
import math

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CUSTOMCFG  # isort: skip
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.tocabi import Tocabi_CFG

from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab.envs import ManagerBasedRLEnvCfg



@configclass
class TocabiActionsCfg:
    joint_pos = mdp.TocabiActionCfg(
        asset_name="robot", 
        clip = {".*": (-1.0, 1.0)},
        lower_joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                           "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"],
        upper_joint_names=["Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
                           "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                           "Neck_Joint", "Head_Joint",
                           "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"],
        pd_control=True,

        p_gains = [2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                   6000.0, 10000.0, 10000.0, 
                   400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0, 
                   100.0, 100.0, 
                   400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0],

        d_gains = [15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                   15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                   200.0, 100.0, 100.0, 
                   10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 
                   3.0, 3.0, 
                   10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0],

        torque_limits= [333, 232, 263, 289, 222, 166,
                        333, 232, 263, 289, 222, 166,
                        303, 303, 303,
                        64, 64, 64, 64, 23, 23, 10, 10,
                        10, 10,
                        64, 64, 64, 64, 23, 23, 10, 10],


        joint_pos_limits = [(-0.3, 0.3), (-0.5, 0.5), (-1.0, 0.5), (-0.3, 1.2), (-0.8, 0.5), (-0.6, 0.6), 
                            (-0.3, 0.3), (-0.5, 0.5), (-1.0, 0.5), (-0.3, 1.2), (-0.8, 0.5), (-0.6, 0.6)],

    )


@configclass
class TocabiCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.5), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
        ),
    )
    phase_time = mdp.WalkingPhaseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.WalkingPhaseCommandCfg.Ranges(
            # phase_time=(1.0, 2.0)
            phase_time=(1.0, 1.0)
        ),
    )
    first_foot_step = mdp.FirstFootStepCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        is_rfoot_first=1.0,
    )

@configclass
class TocabiObservationsCfg:

    @configclass
    class StudentCfg(ObsGroup):
        """ observations for student policy"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        clock_input = ObsTerm(func=mdp.clock_input, params={"command_name": "phase_time", "is_rfoot_first": "first_foot_step"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class TeacherCfg(ObsGroup):
        """ observations for teacher policy"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        clock_input = ObsTerm(func=mdp.clock_input, params={"command_name": "phase_time", "is_rfoot_first": "first_foot_step"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        # actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()

@configclass
class TocabiTeacherObservationsCfg(TocabiObservationsCfg):

    @configclass
    class TeacherCfg(ObsGroup):
        """ observations for teacher policy"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        clock_input = ObsTerm(func=mdp.clock_input, params={"command_name": "phase_time", "is_rfoot_first": "first_foot_step"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        # actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: TeacherCfg = TeacherCfg()

@configclass
class TocabiEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_actuator_properties = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_joint_properties = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )

    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "force_range": (-100.0, 100.0),
            # "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),    
            "asset_cfg": SceneEntityCfg("robot", body_names=["Pelvis_Link"]),
        },
    )

@configclass
class TocabiTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Pelvis_Link"), "threshold": 1.0},
    )
    root_bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.79, "asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TocabiScenceCfg(InteractiveSceneCfg):

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CUSTOMCFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = Tocabi_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors (height_scanner, contact_sensor)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Pelvis_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class Tocabi_WordlModelRewards:
    lin_vel_xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_base_frame_exp, weight=1.0, params={"command_name": "base_velocity", "omega": 5.0})
    ang_vel_z_tracking = RewTerm(func=mdp.track_ang_vel_z_world_exp_tocabi, weight=1.0, params={"command_name": "base_velocity", "omega": 7.0})
    orientation_tracking = RewTerm(func=mdp.orientation_tracking, weight=1.0, params={"omega": 5.0})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    periodic_force = RewTerm(func=mdp.periodic_force, weight=1.0, params={
                        "scale": 1300, 
                        "command_name": "phase_time",
                        "is_rfoot_first": "first_foot_step",
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"
    })
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    feet_height_tracking = RewTerm(func=mdp.feet_height_tracking, weight=1.0, params={
                        # "omega": 5.0, "foot_height": 0.1, "kappa": 4.0, "offset": 0.1585,
                        "omega": 5.0, "foot_height": 0.25, "kappa": 2.0, "offset": -0.77,
                        "command_name": "phase_time", "is_rfoot_first": "first_foot_step",
                        "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    feet_velocity_z_tracking = RewTerm(func=mdp.feet_velocity_z_tracking, weight=0.5, params={
                        # "omega": 5.0, "foot_height": 0.1, "kappa": 4.0, "offset": 0.1585,
                        "omega": 3.0, "foot_height": 0.25, "kappa": 2.0,
                        "command_name": "phase_time", "is_rfoot_first": "first_foot_step",
                        "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    large_contact = RewTerm(func=mdp.large_contact_force, weight=-0.01, params={"threshold": 100.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "asset_cfg": SceneEntityCfg("robot")})
    default_joint = RewTerm(func=mdp.default_joint_pos, weight=0.5, params={"omega": 2.0, 
                        "asset_cfg": SceneEntityCfg("robot",
                            joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                         "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    feet_flat = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"])})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class TocabiTeacherEnvCfg(ManagerBasedRLEnvCfg):
    scene: TocabiScenceCfg = TocabiScenceCfg(num_envs=4096, env_spacing=2.5)
    actions: TocabiActionsCfg = TocabiActionsCfg()
    commands: TocabiCommandsCfg = TocabiCommandsCfg()
    observations: TocabiTeacherObservationsCfg = TocabiTeacherObservationsCfg()
    events: TocabiEventCfg = TocabiEventCfg()
    terminations: TocabiTerminations = TocabiTerminations()
    rewards: Tocabi_WordlModelRewards = Tocabi_WordlModelRewards()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):    
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class TocabiStudentEnvCfg(TocabiTeacherEnvCfg):
    observations: TocabiObservationsCfg = TocabiObservationsCfg()

    def __post_init__(self):
        self.rewards = None 