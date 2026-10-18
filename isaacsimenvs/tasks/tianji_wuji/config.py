"""Single right Tianji arm + Wuji hand; independent of Kuka/Sharpa defaults."""
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg, AssetsCfg
from .assets import DEFAULT_ARM_POSE


@configclass
class TianjiWujiRobotCfg:
    dextele_root: str = ""
    root_pos: tuple[float, float, float] = (-0.48634, 0.28420, 1.01)
    root_quat_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)
    arm_default_joint_pos: tuple[float, ...] = DEFAULT_ARM_POSE
    hand_default_joint_pos: tuple[float, ...] = (0.05,) + (0.,) * 19
    arm_stiffness: tuple[float, ...] = (80., 80., 80., 80., 40., 40., 40.)
    arm_damping: tuple[float, ...] = (8., 8., 8., 8., 4., 4., 4.)
    hand_stiffness: float = 0.5
    hand_damping: float = 0.01
    hand_armature: float = 0.001
    disable_gravity: bool = True
    self_collision: bool = True


@configclass
class TianjiWujiEnvCfg(SimToolRealEnvCfg):
    action_space: int = 27
    observation_space: int = 134
    state_space: int = 156
    robot: TianjiWujiRobotCfg = TianjiWujiRobotCfg()
    # Generated from DexTele at launch; never loads the original robot URDF.
    assets: AssetsCfg = AssetsCfg(robot_urdf="", num_assets_per_type=10)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=2.0,
        replicate_physics=False, clone_in_fabric=False,
    )

    def __post_init__(self):
        super().__post_init__()
        # The hand starts above the table; smaller reset noise avoids moving
        # the long arm through the tabletop at episode initialization.
        self.reset.reset_dof_pos_random_interval_arm = 0.005
        self.reset.reset_dof_pos_random_interval_fingers = 0.05
        self.student_obs.raycast_dynamic_prim_exprs = (
            "/World/envs/env_.*/Table/box/visuals",
            "/World/envs/env_.*/Object/.*/visuals",
            "/World/envs/env_.*/Robot/.*/visuals",
        )
