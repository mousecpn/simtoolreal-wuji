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
    arm_stiffness: tuple[float, ...] = (600., 600., 500., 400., 200., 200., 200.)
    arm_damping: tuple[float, ...] = (8., 8., 8., 8., 4., 4., 4.)
    # DexTele wuji-mjlab Hand 1 right_mjlab.xml, finger-major joint order.
    hand_stiffness: tuple[float, ...] = (
        0.40844710645048043, 0.6858601063643346, 0.2391112196099482, 0.20736128319711747,
        0.37352218155860073, 0.45592448909027794, 0.24368366649522863, 0.18026971340925335,
        0.3687093483646485, 0.4164253443634641, 0.22218607502059182, 0.19427606072023446,
        0.35718151495111794, 0.42977315313086895, 0.24930151196247122, 0.2285032688178066,
        0.3655325975433942, 0.41393113081120425, 0.22729367621965954, 0.1964723550341816,
    )
    restrict_abduction: bool = True
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
    # Generated from bundled Tianji/Wuji sources at launch.
    assets: AssetsCfg = AssetsCfg(robot_urdf="", num_assets_per_type=100)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=24576, env_spacing=2.0,
        replicate_physics=False, clone_in_fabric=False,
    )

    def __post_init__(self):
        super().__post_init__()
        # DexTele scene table top: 0.64 m = table body z 0.49 + half-height 0.15.
        self.reset.table_reset_z = 0.49
        self.reset.reset_dof_pos_random_interval_arm = 0.1
        self.reset.reset_dof_pos_random_interval_fingers = 0.1
        self.termination.force_consecutive_near_goal_steps = True
        self.student_obs.raycast_dynamic_prim_exprs = (
            "/World/envs/env_.*/Table/box/visuals",
            "/World/envs/env_.*/Object/.*/visuals",
            "/World/envs/env_.*/Robot/.*/visuals",
        )
