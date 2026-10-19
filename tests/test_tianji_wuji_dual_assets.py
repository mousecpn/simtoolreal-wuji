"""Dual robot source fidelity and backward-compatible single-mode checks."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

MODULE = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/tianji_wuji/assets.py"
SPEC = importlib.util.spec_from_file_location("dual_asset_tests", MODULE)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class DualAssetTests(unittest.TestCase):
    def test_dual_tree_inertias_contacts_and_portable_meshes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = assets.prepare_asset(output_dir=directory, arms="dual")
            robot = ET.parse(path).getroot()
            joints = [j.get("name") for j in robot.findall("joint") if j.get("type") == "revolute"]
            self.assertEqual(tuple(joints), assets.DUAL_DOF_NAMES)
            self.assertEqual(len(joints), 54)
            manifest = json.loads((path.parent / "manifest.json").read_text())
            for side, palm in zip(("left", "right"), assets.DUAL_PALM_LINKS):
                native = ET.parse(manifest["palm_inertia_sources"][side])
                source = native.find(f"link[@name='{side}_palm_link']/inertial")
                converted = robot.find(f"link[@name='{palm}']/inertial")
                self.assertEqual([(e.tag, e.attrib) for e in converted.iter()],
                                 [(e.tag, e.attrib) for e in source.iter()])
                self.assertEqual(len(manifest["palm_inertia_sources_sha256"][side]), 64)
                self.assertTrue(any(set(pair) == {palm, f"{side}_finger2_link2"}
                                    for pair in manifest["contact_exclusions"]))
            for tip in assets.DUAL_FINGERTIP_LINKS:
                self.assertIsNotNone(robot.find(f"link[@name='{tip}']/collision"))
                self.assertIsNotNone(robot.find(f"joint[@name='{tip}_fixed'][@type='fixed']"))
                self.assertEqual(manifest["collision_masks"][tip], [[1, 1]])
            for mesh in robot.findall(".//mesh"):
                filename = Path(mesh.get("filename"))
                self.assertFalse(filename.is_absolute())
                self.assertTrue((path.parent / filename).is_file())
            # Default pose mirrors the unchanged single-arm placement.
            fk = assets.forward_kinematics(path)
            left = [row[3] for row in fk[assets.LEFT_PALM_LINK]][:3]
            right = [row[3] for row in fk[assets.PALM_LINK]][:3]
            for a, b in zip(left, (right[0], -right[1], right[2])):
                self.assertAlmostEqual(a, b, delta=1e-5)

    def test_single_default_does_not_change_when_dual_is_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = assets.prepare_asset(output_dir=root / "single")
            before = default.read_bytes()
            manifest = (default.parent / "manifest.json").read_bytes()
            assets.prepare_asset(output_dir=root / "dual", arms="dual")
            explicit = assets.prepare_asset(output_dir=root / "explicit", arms="single")
            self.assertEqual(before, explicit.read_bytes())
            self.assertEqual(manifest, (explicit.parent / "manifest.json").read_bytes())
            self.assertNotIn(b'left_wuji_palm', before)
            self.assertEqual(default.name, "tianji_wuji_right.urdf")
            with self.assertRaisesRegex(ValueError, "arms must be"):
                assets.prepare_asset(output_dir=root / "bad", arms="both")

    def test_runtime_default_joint_positions_respect_source_limits(self):
        # Read the actual config values without importing Isaac Sim. Compile
        # just these assignments, with their imported asset constants supplied.
        config = MODULE.parents[1] / "tianji_wuji_dual/config.py"
        tree = ast.parse(config.read_text())
        robot = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "TianjiWujiDualRobotCfg")
        namespace = {"LEFT_DEFAULT_ARM_POSE": assets.LEFT_DEFAULT_ARM_POSE,
                     "DEFAULT_ARM_POSE": assets.DEFAULT_ARM_POSE}
        fields = {"arm_default_joint_pos", "hand_default_joint_pos"}
        for node in robot.body:
            if isinstance(node, ast.AnnAssign) and node.target.id in fields:
                expression = ast.Expression(body=node.value)
                namespace[node.target.id] = eval(compile(expression, str(config), "eval"),
                                                {"__builtins__": {}}, namespace)
        arm_names = assets.LEFT_ARM_JOINT_NAMES + assets.ARM_JOINT_NAMES
        hand_names = assets.LEFT_HAND_JOINT_NAMES + assets.HAND_JOINT_NAMES
        defaults = {}
        for names, field in ((arm_names, "arm_default_joint_pos"),
                             (hand_names, "hand_default_joint_pos")):
            self.assertEqual(len(names), len(namespace[field]))
            defaults.update(zip(names, namespace[field]))
        with tempfile.TemporaryDirectory() as directory:
            robot = ET.parse(assets.prepare_asset(output_dir=directory, arms="dual"))
            for joint in robot.findall("joint[@type='revolute']"):
                name = joint.get("name")
                limit = joint.find("limit")
                with self.subTest(joint=name):
                    self.assertGreaterEqual(defaults[name], float(limit.get("lower")))
                    self.assertLessEqual(defaults[name], float(limit.get("upper")))

    def test_independent_dual_fk_matches_canonical_mjcf(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from yourdfpy import URDF
        source = ET.parse(assets.find_source()).find("worldbody/body[@name='base']")
        with tempfile.TemporaryDirectory() as directory:
            urdf = URDF.load(assets.prepare_asset(output_dir=directory, arms="dual"), load_meshes=False)
            rng = np.random.default_rng(143)
            for _ in range(6):
                q = {j.get("name"): rng.uniform(*map(float, j.get("range").split()))
                     for j in source.iter("joint")}
                urdf.update_cfg(q)

                def check(body, parent):
                    local = np.eye(4)
                    if body is not source:
                        local[:3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
                        if body.get("quat"):
                            w, x, y, z = map(float, body.get("quat").split())
                            local[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
                        elif body.get("xyaxes"):
                            x, y = np.fromstring(body.get("xyaxes"), sep=" ").reshape(2, 3)
                            x /= np.linalg.norm(x)
                            y -= x * np.dot(x, y)
                            y /= np.linalg.norm(y)
                            local[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
                    joint = body.find("joint")
                    if joint is not None:
                        rotation = np.eye(4)
                        axis = np.fromstring(joint.get("axis"), sep=" ")
                        rotation[:3, :3] = Rotation.from_rotvec(axis * q[joint.get("name")]).as_matrix()
                        local = local @ rotation
                    world = parent @ local
                    np.testing.assert_allclose(urdf.get_transform(body.get("name")), world, atol=2e-10)
                    for child in body.findall("body"):
                        check(child, world)

                check(source, np.eye(4))


if __name__ == "__main__":
    unittest.main()
