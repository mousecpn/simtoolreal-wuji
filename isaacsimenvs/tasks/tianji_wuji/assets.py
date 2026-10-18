"""Prepare an Isaac Sim URDF from DexTele's canonical Tianji/Wuji MJCF.

No simulator dependency is needed.  Meshes are copied into the generated asset,
so the resulting URDF remains usable without the source checkout.  The torso
frame is the URDF origin; its world mounting height belongs to the environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ARM_JOINT_NAMES = tuple(f"Joint{i}_R" for i in range(1, 8))
HAND_JOINT_NAMES = tuple(f"right_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5))
DOF_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES
DEFAULT_ARM_POSE = (-0.712, -0.963, 0.0, -0.85, -0.356, 0.0, 0.0)
PALM_LINK = "right_wuji_palm"
FINGERTIP_LINKS = tuple(f"right_finger{i}_tip_link" for i in range(1, 6))
SOURCE_RELATIVE = Path("robots/tianji/tianji_marvin_CCS_real_limit_wuji_sim_infra.xml")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _numbers(value, default):
    return tuple(float(v) for v in value.split()) if value else default


def _fmt(values):
    return " ".join(f"{v:.12g}" for v in values)


def _rotation(element):
    """Return a rotation matrix for the orientation forms used by this source."""
    if element.get("xyaxes"):
        axes = _numbers(element.get("xyaxes"), ())
        x, y = axes[:3], axes[3:]
        norm = lambda v: tuple(a / math.sqrt(sum(b*b for b in v)) for a in v)
        x = norm(x)
        dot = sum(a*b for a, b in zip(x, y))
        y = norm(tuple(a-dot*b for a, b in zip(y, x)))
        z = (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
        return tuple(tuple(v[i] for v in (x, y, z)) for i in range(3))
    w, x, y, z = _numbers(element.get("quat"), (1., 0., 0., 0.))
    n = math.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = (v/n for v in (w, x, y, z))
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def _rpy(matrix):
    pitch = math.asin(max(-1., min(1., -matrix[2][0])))
    if abs(math.cos(pitch)) < 1e-8:
        return (math.atan2(-matrix[1][2], matrix[1][1]), pitch, 0.)
    return (math.atan2(matrix[2][1], matrix[2][2]), pitch, math.atan2(matrix[1][0], matrix[0][0]))


def _origin(parent, source):
    ET.SubElement(parent, "origin", xyz=source.get("pos", "0 0 0"), rpy=_fmt(_rpy(_rotation(source))))


def find_source(dextele_root=None):
    root = Path(dextele_root or os.environ.get("DEXTELE_ROOT", REPO_ROOT.parent / "DexTele")).expanduser()
    source = root / SOURCE_RELATIVE
    if not source.is_file():
        raise FileNotFoundError(f"Tianji/Wuji source not found: {source}. Set env.robot.dextele_root or DEXTELE_ROOT to the DexTele checkout containing {SOURCE_RELATIVE}.")
    return source


def prepare_asset(dextele_root=None, output_dir=None):
    """Generate a single right arm + right hand URDF, returning its path.

    Joint frames, mesh scales, principal inertia frames, efforts, and limits are
    copied from the canonical MJCF.  The palm inertia comes from Wuji's native
    URDF because the MJCF leaves its inertia to mesh-density inference.  The
    source collision-enabled mesh set is retained. Pairwise MJCF exclusions are
    recorded in the manifest for the Isaac Sim scene's USD collision filters.
    """
    source = find_source(dextele_root)
    target = Path(output_dir or REPO_ROOT / "assets/tianji_wuji").expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    mesh_target = target / "meshes"
    mesh_target.mkdir(exist_ok=True)
    mj = ET.parse(source).getroot()
    meshdir = source.parent / mj.find("compiler").get("meshdir", "")
    meshes = {m.get("name"): m for m in mj.findall("asset/mesh")}
    materials = {m.get("name"): m.get("rgba", "0.8 0.8 0.8 1") for m in mj.findall("asset/material")}
    defaults = {d.get("class"): d.find("joint").attrib for d in mj.findall("default/default") if d.find("joint") is not None}
    urdf = ET.Element("robot", name="tianji_wuji_right")
    base = mj.find("worldbody/body[@name='base']")
    if base is None or base.find("body[@name='Link1_R']") is None:
        raise ValueError(f"Expected canonical base/Link1_R hierarchy in {source}")
    joints = []
    mesh_sources = {}
    collision_masks = {}
    native = source.parent / "wuji_hand_sim_infra/urdf/right.urdf"

    def mesh_geometry(parent, geom):
        mesh = meshes[geom.get("mesh")]
        # MJCF resolves '..' lexically before following meshdir symlinks.
        resolved = Path(os.path.normpath(meshdir / mesh.get("file"))).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing DexTele mesh {resolved} referenced by {source}. Restore DexTele's meshes_sim_infra and wuji_hand_sim_infra symlinks and the sibling sim_infra asset checkout.")
        filename = f"{mesh.get('name')}{resolved.suffix}"
        shutil.copy2(resolved, mesh_target / filename)
        mesh_sources[str(resolved)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        geometry = ET.SubElement(parent, "geometry")
        ET.SubElement(geometry, "mesh", filename=f"meshes/{filename}", scale=mesh.get("scale", "1 1 1"))

    def visit(body, parent_name=None, inherited_class=None):
        name = body.get("name")
        link = ET.SubElement(urdf, "link", name=name)
        inertia = body.find("inertial")
        if inertia is not None:
            output = ET.SubElement(link, "inertial")
            _origin(output, inertia)
            ET.SubElement(output, "mass", value=inertia.get("mass"))
            diagonal = _numbers(inertia.get("diaginertia"), ())
            ET.SubElement(output, "inertia", ixx=str(diagonal[0]), iyy=str(diagonal[1]), izz=str(diagonal[2]), ixy="0", ixz="0", iyz="0")
        elif name == PALM_LINK:
            # Native hand palm inertia, expressed in its mesh/body frame.
            if not native.is_file():
                raise FileNotFoundError(f"Missing native Wuji palm inertia source: {native}")
            palm_inertia = ET.parse(native).find("link[@name='right_palm_link']/inertial")
            link.append(palm_inertia)
        if parent_name:
            joint = body.find("joint")
            output = ET.SubElement(urdf, "joint", name=joint.get("name") if joint is not None else f"{name}_fixed", type="revolute" if joint is not None else "fixed")
            ET.SubElement(output, "parent", link=parent_name)
            ET.SubElement(output, "child", link=name)
            _origin(output, body)
            if joint is not None:
                if _numbers(joint.get("pos"), (0., 0., 0.)) != (0., 0., 0.):
                    raise ValueError("Nonzero MJCF joint origins require an intermediate URDF frame")
                ET.SubElement(output, "axis", xyz=joint.get("axis", "0 0 1"))
                low, high = _numbers(joint.get("range"), ())
                force = _numbers(joint.get("actuatorfrcrange"), (-1., 1.))
                velocity = 3.14 if joint.get("name") in ARM_JOINT_NAMES else 8.58
                ET.SubElement(output, "limit", lower=str(low), upper=str(high), effort=str(max(abs(v) for v in force)), velocity=str(velocity))
                attrs = dict(defaults.get(joint.get("class", inherited_class), {}))
                attrs.update(joint.attrib)
                ET.SubElement(output, "dynamics", damping=attrs.get("damping", "0"), friction=attrs.get("frictionloss", "0"))
                joints.append({"name": joint.get("name"), "lower": low, "upper": high, "effort": max(abs(v) for v in force), "velocity": velocity, "armature": float(attrs.get("armature", 0))})
        # A single visual per geometry, plus only the collision-enabled copy.
        seen_visuals = set()
        for geom in body.findall("geom"):
            if geom.get("mesh") == "Base_L":
                continue
            if geom.get("mesh", "").endswith("_tip_link"):
                continue  # Attached to a dedicated fixed tip body below.
            if geom.get("type", "mesh") != "mesh":
                raise ValueError("Only mesh geometry is expected in the canonical robot")
            key = (geom.get("mesh"), geom.get("pos"), geom.get("quat"))
            if key not in seen_visuals:
                visual = ET.SubElement(link, "visual")
                _origin(visual, geom)
                mesh_geometry(visual, geom)
                material = ET.SubElement(visual, "material", name=f"{name}_color")
                ET.SubElement(material, "color", rgba=geom.get("rgba", materials.get(geom.get("material"), "0.8 0.8 0.8 1")))
                seen_visuals.add(key)
            if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
                collision_masks.setdefault(name, []).append((int(geom.get("contype", "1")), int(geom.get("conaffinity", "1"))))
                collision = ET.SubElement(link, "collision")
                _origin(collision, geom)
                mesh_geometry(collision, geom)
        for child in body.findall("body"):
            if child.get("name") == "Link1_L":
                continue
            visit(child, name, body.get("childclass", inherited_class))
        # Preserve tip frames as real fixed URDF links for contact observations.
        if name.startswith("right_finger") and name.endswith("_link4"):
            tip = name.replace("_link4", "_tip_link")
            tip_geom = next(g for g in body.findall("geom") if g.get("mesh") == tip)
            tip_link = ET.SubElement(urdf, "link", name=tip)
            # Negligible explicit inertia keeps this contact body loadable with
            # fixed-joint collapsing disabled; tip mass is in source link4.
            inertial = ET.SubElement(tip_link, "inertial")
            ET.SubElement(inertial, "mass", value="0.000001")
            ET.SubElement(inertial, "inertia", ixx="1e-10", iyy="1e-10", izz="1e-10", ixy="0", ixz="0", iyz="0")
            for kind in ("visual", "collision"):
                geometry_parent = ET.SubElement(tip_link, kind)
                mesh_geometry(geometry_parent, tip_geom)
            fixed = ET.SubElement(urdf, "joint", name=f"{tip}_fixed", type="fixed")
            ET.SubElement(fixed, "parent", link=name)
            ET.SubElement(fixed, "child", link=tip)
            _origin(fixed, tip_geom)

    visit(base)
    if tuple(j["name"] for j in joints) != DOF_NAMES:
        raise ValueError(f"Unexpected joint ordering: {[j['name'] for j in joints]}")
    ET.indent(urdf)
    path = target / "tianji_wuji_right.urdf"
    ET.ElementTree(urdf).write(path, encoding="utf-8", xml_declaration=True)
    names = {link.get("name") for link in urdf.findall("link")}
    exclusions = [(c.get("body1"), c.get("body2")) for c in mj.findall("contact/exclude") if c.get("body1") in names and c.get("body2") in names]
    # URDF cannot express MuJoCo collision bit masks. Convert incompatible
    # robot body masks to pair exclusions consumed by the USD scene builder.
    masked_bodies = list(collision_masks)
    for index, first in enumerate(masked_bodies):
        for second in masked_bodies[index + 1:]:
            compatible = any((type_a & affinity_b) or (type_b & affinity_a)
                             for type_a, affinity_a in collision_masks[first]
                             for type_b, affinity_b in collision_masks[second])
            if not compatible and (first, second) not in exclusions:
                exclusions.append((first, second))
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mesh_sources_sha256": mesh_sources,
        "palm_inertia_source": str(native.resolve()),
        "palm_inertia_source_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "attribution": {
            "robot_model": "Tianji Marvin CCS-696 arm and Wuji right hand, canonical sim_infra model distributed through DexTele",
            "derivation": "Right chain extracted from canonical MJCF; native Wuji URDF supplies palm inertia; contact tips promoted to fixed bodies with 1e-6 kg inertia placeholders.",
            "license": "Source asset licensing remains with its original providers; this conversion does not relicense the source models or meshes.",
        },
        "joints": joints, "base_link": "base", "palm_link": PALM_LINK,
        "fingertip_links": FINGERTIP_LINKS, "default_arm_pose": DEFAULT_ARM_POSE,
        "self_collisions": "Configured by env.robot.self_collision; scene applies contact_exclusions and adjacent-link filters.",
        "contact_exclusions": exclusions,
        "collision_masks": collision_masks,
        "velocity_limits": "Conservative configured caps: arm 3.14 rad/s, hand 8.58 rad/s; canonical MJCF does not specify velocity limits.",
        "world_mount": "Environment-controlled; canonical MJCF base z=0.87 deliberately omitted.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


prepare_tianji_wuji_asset = prepare_asset


def forward_kinematics(urdf_path, positions=None):
    """Return body transforms in the unmounted torso frame (stdlib only).

    Useful for placing task objects near the hand without hardcoded world
    offsets. Values are homogeneous 4-by-4 matrices, keyed by URDF link name.
    """
    positions = dict(zip(ARM_JOINT_NAMES, DEFAULT_ARM_POSE)) if positions is None else positions
    robot = ET.parse(urdf_path).getroot()
    identity = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))
    multiply = lambda a, b: tuple(tuple(sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)) for i in range(4))
    children = {j.find("child").get("link") for j in robot.findall("joint")}
    result = {link.get("name"): identity for link in robot.findall("link") if link.get("name") not in children}
    pending = list(robot.findall("joint"))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in result:
                continue
            origin = joint.find("origin")
            xyz = _numbers(origin.get("xyz"), (0., 0., 0.))
            roll, pitch, yaw = _numbers(origin.get("rpy"), (0., 0., 0.))
            cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
            rotation = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr), (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))
            transform = tuple(tuple(rotation[i][j] if j < 3 else xyz[i] for j in range(4)) for i in range(3)) + ((0., 0., 0., 1.),)
            if joint.get("type") == "revolute":
                axis = _numbers(joint.find("axis").get("xyz"), (0., 0., 1.))
                angle = positions.get(joint.get("name"), 0.)
                # Source thumb resting pose is slightly above zero.
                limit = joint.find("limit")
                angle = max(float(limit.get("lower")), min(float(limit.get("upper")), angle))
                x, y, z = axis
                c, s, v = math.cos(angle), math.sin(angle), 1-math.cos(angle)
                rot = ((x*x*v+c, x*y*v-z*s, x*z*v+y*s, 0.), (y*x*v+z*s, y*y*v+c, y*z*v-x*s, 0.), (z*x*v-y*s, z*y*v+x*s, z*z*v+c, 0.), (0., 0., 0., 1.))
                transform = multiply(transform, rot)
            result[joint.find("child").get("link")] = multiply(result[parent], transform)
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("URDF joints contain a disconnected or cyclic hierarchy")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dextele-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(prepare_asset(args.dextele_root, args.output_dir))
