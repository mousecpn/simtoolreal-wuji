"""Independent scene setup for one Tianji right arm and one Wuji right hand.

Only robot-specific setup lives here. Object generation, conversion, table assets,
and optional cameras use the existing task's generic utilities without changing
its configuration or module globals.
"""
from __future__ import annotations

import json
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane

from ..simtoolreal.utils.scene_utils import (
    _bake_usd, _build_object_scale_tensor, _convert_urdf_to_usd,
    _generate_scaled_table_urdfs, _log_scene_step, _materialize_env_prims,
    _robot_joint_drive_cfg, _prepare_urdf_for_isaacsim, build_rigid_object_cfg,
    hide_goal_viz_for_student_camera, setup_student_camera,
)
from ..simtoolreal.utils.generate_objects import generate_handle_head_urdfs
from .assets import prepare_asset, PALM_LINK, FINGERTIP_LINKS

ARM_JOINT_REGEX = r"Joint[1-7]_R"
HAND_JOINT_REGEX = r"right_finger[1-5]_joint[1-4]"
ARM_JOINT_NAMES = tuple(f"Joint{i}_R" for i in range(1, 8))
HAND_JOINT_NAMES = tuple(f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
JOINT_NAMES_CANONICAL = ARM_JOINT_NAMES + HAND_JOINT_NAMES
PALM_BODY_NAME = PALM_LINK
FINGERTIP_LINK_NAMES = FINGERTIP_LINKS
FINGERTIP_BODY_REGEX = r"right_finger[1-5]_tip_link"


def _joint_values(value, names):
    """Allow a shared gain, canonical sequence, or explicit per-joint map."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (tuple, list)):
        if len(value) != len(names):
            raise ValueError(f"Expected {len(names)} joint values, got {len(value)}")
        return dict(zip(names, map(float, value)))
    return float(value)


def build_robot_articulation_usd_cfg(usd_path: str, robot_cfg) -> ArticulationCfg:
    arm_pose = tuple(robot_cfg.arm_default_joint_pos)
    hand_pose = tuple(robot_cfg.hand_default_joint_pos)
    if len(arm_pose) != 7 or len(hand_pose) != 20:
        raise ValueError("Tianji/Wuji requires 7 arm and 20 hand initial joint positions")
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(robot_cfg.root_pos), rot=tuple(robot_cfg.root_quat_wxyz),
            joint_pos=dict(zip(JOINT_NAMES_CANONICAL, arm_pose + hand_pose)),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[ARM_JOINT_REGEX],
                stiffness=_joint_values(robot_cfg.arm_stiffness, ARM_JOINT_NAMES),
                damping=_joint_values(robot_cfg.arm_damping, ARM_JOINT_NAMES),
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[HAND_JOINT_REGEX],
                stiffness=_joint_values(robot_cfg.hand_stiffness, HAND_JOINT_NAMES),
                damping=_joint_values(robot_cfg.hand_damping, HAND_JOINT_NAMES),
                armature=_joint_values(robot_cfg.hand_armature, HAND_JOINT_NAMES),
            ),
        },
    )


def _apply_robot_collision_filters(usd_path: str, urdf_path: Path) -> None:
    """Filter URDF neighbors and canonical DexTele contact exclusions.

    Fixed links stay independent, including Wuji palm and contact tip bodies.
    """
    from pxr import Usd, UsdPhysics

    root = ET.parse(urdf_path).getroot()
    edges = [(joint.find("parent").get("link"), joint.find("child").get("link"))
             for joint in root.findall("joint")]
    manifest_path = urdf_path.parent / "manifest.json"
    exclusions = json.loads(manifest_path.read_text()).get("contact_exclusions", [])
    # MJCF tip geometries belonged to link4; preserve its exclusions after
    # promoting each contact tip to a fixed body for observations/materials.
    def collision_aliases(name):
        if name.startswith("right_finger") and name.endswith("_link4"):
            return (name, name.replace("_link4", "_tip_link"))
        return (name,)

    pairs = set()
    for first, second in [*edges, *exclusions]:
        pairs.update((a, b) for a in collision_aliases(first)
                     for b in collision_aliases(second) if a != b)

    path = Path(usd_path)
    physics_path = path.parent / "configuration" / f"{path.stem}_physics.usd"
    edit_path = physics_path if physics_path.exists() else path
    stage = Usd.Stage.Open(str(edit_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Cannot open robot USD: {edit_path}")
    stage.Load()
    bodies = {p.GetName(): p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    for a, b in pairs:
        if a in bodies and b in bodies:
            UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
    stage.GetRootLayer().Save()


def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(
        setup_t0,
        f"setup start num_envs={env.num_envs} "
        f"num_assets_per_type={assets_cfg.num_assets_per_type}",
    )

    # 1. Resolve the object pool: a single named URDF (DexToolBench eval) or
    #    procedural URDFs generated in a per-launch temp dir.
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="tianji_wuji_assets_")
    if assets_cfg.object_urdf:
        if assets_cfg.object_scale is None:
            raise ValueError(
                "cfg.assets.object_scale must be set when object_urdf is given "
                "(policy-normalized grasp-bbox scale, NAME_TO_OBJECT convention)."
            )
        urdf_paths = [Path(assets_cfg.object_urdf)]
        if not urdf_paths[0].exists():
            raise FileNotFoundError(f"object_urdf not found: {urdf_paths[0]}")
        object_scales_normalized = [tuple(assets_cfg.object_scale)]
    else:
        urdf_paths, object_scales_normalized = generate_handle_head_urdfs(
            handle_head_types=tuple(assets_cfg.handle_head_types),
            num_per_type=assets_cfg.num_assets_per_type,
            out_dir=env._tmp_asset_dir,
            shuffle=assets_cfg.shuffle_assets,
        )
    if not urdf_paths:
        raise ValueError(
            "No procedural object URDFs were generated. "
            "Check cfg.assets.handle_head_types and num_assets_per_type."
        )
    env._object_urdf_paths = [str(path) for path in urdf_paths]
    _log_scene_step(setup_t0, f"generated {len(urdf_paths)} object URDFs")

    # 2. Convert URDFs -> raw USDs -> role-specific baked USDs.
    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    object_raw_usds = [
        _convert_urdf_to_usd(
            str(urdf), usd_work_dir, fix_base=False, replace_cylinders_with_capsules=True,
        )
        for urdf in urdf_paths
    ]
    object_usd_paths = [
        _bake_usd(usd, bake_root, "object", props=dict(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1000.0, articulation_enabled=False,
        ))
        for usd in object_raw_usds
    ]
    goalviz_usd_paths = [
        _bake_usd(usd, bake_root, "goalviz", props=dict(
            kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
        ), collision_enabled=False)
        for usd in object_raw_usds
    ]

    # Build a new, single-right-arm URDF; the source assets stay untouched.
    robot_urdf = prepare_asset(
        dextele_root=getattr(env.cfg.robot, "dextele_root", None),
        output_dir=Path(env._tmp_asset_dir) / "robot",
    )
    env._robot_urdf_path = str(robot_urdf)
    # Preserve palm and contact tip fixed links for observation/contact identity.
    robot_converted_usd = UrdfConverter(UrdfConverterCfg(
        asset_path=_prepare_urdf_for_isaacsim(str(robot_urdf), usd_work_dir),
        usd_dir=str(usd_work_dir / "tianji_wuji_robot"),
        force_usd_conversion=True, fix_base=True, merge_fixed_joints=False,
        make_instanceable=False,
        self_collision=bool(env.cfg.robot.self_collision),
        joint_drive=_robot_joint_drive_cfg(),
    )).usd_path
    if env.cfg.robot.self_collision:
        _apply_robot_collision_filters(robot_converted_usd, robot_urdf)
    robot_usd_path = _bake_usd(
        robot_converted_usd, bake_root, "robot",
        props=dict(
            disable_gravity=bool(env.cfg.robot.disable_gravity),
            max_depenetration_velocity=1000.0,
            enabled_self_collisions=bool(env.cfg.robot.self_collision),
            solver_position_iterations=8, solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )
    # Table USD(s). When table_scale_range_x/y are non-trivial and
    # table_scale_num_variants > 1, pre-bake N scaled URDF variants and pass
    # them as a list to RigidObject — Isaac Lab's MultiUsdFileCfg cycles
    # through the list, giving each env one of the variants. Z scale is held
    # at 1.0 so the table surface height matches the policy's expectation.
    scale_range_x = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_x", (1.0, 1.0)))
    scale_range_y = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_y", (1.0, 1.0)))
    n_table_variants = int(getattr(assets_cfg, "table_scale_num_variants", 1))
    table_scale_is_trivial = (
        scale_range_x == (1.0, 1.0) and scale_range_y == (1.0, 1.0)
    ) or n_table_variants <= 1
    if table_scale_is_trivial:
        table_usd_paths = [_bake_usd(
            _convert_urdf_to_usd(assets_cfg.table_urdf, usd_work_dir, fix_base=False),
            bake_root, "table",
            props=dict(
                kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
            ),
        )]
        # Single (sx, sy) = (1.0, 1.0) for downstream consumers (eval viz).
        env._table_variant_scales = [(1.0, 1.0)]
    else:
        variant_urdf_dir = Path(env._tmp_asset_dir) / "table_variants"
        variant_urdf_paths, variant_scales = _generate_scaled_table_urdfs(
            base_urdf_path=assets_cfg.table_urdf,
            num_variants=n_table_variants,
            scale_range_x=scale_range_x,
            scale_range_y=scale_range_y,
            out_dir=variant_urdf_dir,
            # Deterministic across runs so the on-disk variants are stable.
            # The env-level seed governs which variant lands in which env via
            # Isaac Lab's round-robin spawn ordering.
            seed=0,
        )
        env._table_variant_scales = list(variant_scales)
        table_usd_paths = [
            _bake_usd(
                _convert_urdf_to_usd(p, usd_work_dir, fix_base=False),
                bake_root, f"table_variant_{idx:03d}",
                props=dict(
                    kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
                ),
            )
            for idx, p in enumerate(variant_urdf_paths)
        ]
        # variant_scales already stashed above for downstream consumers.
        _log_scene_step(
            setup_t0,
            f"baked {len(table_usd_paths)} scaled table USD variants "
            f"x_range={scale_range_x} y_range={scale_range_y}",
        )
    _log_scene_step(setup_t0, "resolved baked USDs")

    # 3. Pre-create env roots so regex spawns resolve to every env.
    _materialize_env_prims(env)

    # 4. Spawn assets.
    env.robot = Articulation(build_robot_articulation_usd_cfg(
        robot_usd_path, env.cfg.robot,
    ))
    env.table = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_paths))
    env.object = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_paths))
    env.goal_viz = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_paths))
    _log_scene_step(setup_t0, "spawned robot/table/object/goalviz")

    # 5. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 6. Per-env scale tensor for spawned Objects.
    _build_object_scale_tensor(env, object_scales_normalized, len(object_usd_paths))

    # 7. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    setup_student_camera(env)
    _log_scene_step(setup_t0, "registered assets with scene")


def apply_physx_material_properties(env) -> None:
    """Assign Wuji tip friction using actual tip body names after conversion."""
    cfg = env.cfg.assets
    if not cfg.modify_asset_frictions:
        return
    dr = env.cfg.domain_randomization
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")
    view = env.robot.root_physx_view
    materials = view.get_material_properties()
    materials[:] = torch.tensor([cfg.robot_friction, cfg.robot_friction, 0.0])
    tip_mask = torch.zeros(view.max_shapes, dtype=torch.bool)
    offset = 0
    found_tips = set()
    for name, path in zip(view.shared_metatype.link_names, view.link_paths[0]):
        count = env.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
        if name in FINGERTIP_LINK_NAMES:
            tip_mask[offset:offset + count] = True
            if count:
                found_tips.add(name)
        offset += count
    if offset != view.max_shapes:
        raise RuntimeError(f"Robot shape count mismatch: {offset} != {view.max_shapes}")
    if found_tips != set(FINGERTIP_LINK_NAMES):
        raise RuntimeError(f"Wuji contact tip collision bodies missing: {set(FINGERTIP_LINK_NAMES) - found_tips}")
    low, high = map(float, dr.fingertip_friction_scale_range)
    buckets = torch.linspace(low, high, int(dr.friction_n_buckets)) * float(cfg.finger_tip_friction)
    scales = buckets[torch.randint(0, len(buckets), (env.num_envs,))]
    materials[:, tip_mask, 0] = scales[:, None]
    materials[:, tip_mask, 1] = scales[:, None]
    view.set_material_properties(materials, env_ids)
    for name in ("table", "object", "goal_viz", "hole"):
        if not hasattr(env, name):
            continue
        obj_view = getattr(env, name).root_physx_view
        obj_materials = obj_view.get_material_properties()
        friction = float(cfg.table_friction if name == "table" else cfg.object_friction)
        obj_materials[:] = torch.tensor([friction, friction, 0.0])
        if name == "object":
            low, high = map(float, dr.object_friction_scale_range)
            buckets = torch.linspace(low, high, int(dr.friction_n_buckets)) * friction
            values = buckets[torch.randint(0, len(buckets), (env.num_envs,))]
            obj_materials[:, :, 0] = values[:, None]
            obj_materials[:, :, 1] = values[:, None]
        obj_view.set_material_properties(obj_materials, env_ids)
