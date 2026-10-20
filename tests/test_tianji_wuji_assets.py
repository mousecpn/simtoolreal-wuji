"""Run without importing Isaac Sim: python -m unittest discover -s tests."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


MODULE = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/tianji_wuji/assets.py"
SPEC = importlib.util.spec_from_file_location("tianji_wuji_assets", MODULE)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def test_default_source_is_bundled(self):
        with patch.dict(os.environ, {"DEXTELE_ROOT": ""}):
            self.assertEqual(assets.find_source(), assets.BUNDLED_ASSET_ROOT / assets.SOURCE_RELATIVE)
            self.assertTrue(assets.find_source().is_file())

    def test_bundle_inventory_and_hashes(self):
        root = assets.BUNDLED_ASSET_ROOT
        manifest = json.loads((root / "MANIFEST.json").read_text())
        inventory = {entry["path"]: entry for entry in manifest["files"]}
        self.assertTrue(inventory)
        actual = set()
        for path in root.rglob("*"):
            self.assertFalse(path.is_symlink(), str(path))
            if path.is_file() and path.name not in {"README.md", "MANIFEST.json"}:
                actual.add(path.relative_to(root).as_posix())
        self.assertEqual(actual, set(inventory))
        for relative, entry in inventory.items():
            with self.subTest(path=relative):
                path = root / relative
                self.assertTrue(path.resolve().is_relative_to(root.resolve()))
                data = path.read_bytes()
                self.assertEqual(len(data), entry["size_bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])

    def test_isolated_checkout_converts_without_external_assets(self):
        """Only a copied loader and bundled payload must suffice in a new root."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "isaacsimenvs/tasks/tianji_wuji/assets.py"
            module.parent.mkdir(parents=True)
            shutil.copy2(MODULE, module)
            bundle = root / "assets/robots/tianji_wuji"
            shutil.copytree(assets.BUNDLED_ASSET_ROOT, bundle)
            spec = importlib.util.spec_from_file_location("isolated_tianji_wuji_assets", module)
            isolated = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(isolated)
            with patch.dict(os.environ, {"DEXTELE_ROOT": ""}):
                self.assertEqual(isolated.find_source(), bundle / assets.SOURCE_RELATIVE)
                urdf = isolated.prepare_asset()
            manifest = json.loads((urdf.parent / "manifest.json").read_text())
            source_paths = [manifest["source"], manifest["palm_inertia_source"],
                            *manifest["mesh_sources_sha256"]]
            self.assertTrue(manifest["mesh_sources_sha256"])
            for source in source_paths:
                self.assertTrue(Path(source).resolve().is_relative_to(bundle), source)
            # The generated robot must remain loadable even after deleting its source.
            shutil.rmtree(bundle)
            for mesh in ET.parse(urdf).findall(".//mesh"):
                relative = Path(mesh.get("filename"))
                self.assertFalse(relative.is_absolute())
                self.assertTrue((urdf.parent / relative).is_file())

    def test_all_bundled_model_references_are_internal(self):
        """Validate both hands and every supplied model, beyond the task's right chain."""
        root = assets.BUNDLED_ASSET_ROOT.resolve()
        references = 0
        for source in root.rglob("*"):
            if source.suffix.lower() not in {".xml", ".urdf"}:
                continue
            model = ET.parse(source).getroot()
            compiler = model.find("compiler")
            meshdir = compiler.get("meshdir", "") if compiler is not None else ""
            texturedir = compiler.get("texturedir", "") if compiler is not None else ""
            for element in model.iter():
                reference = element.get("filename") or element.get("file")
                if not reference:
                    continue
                with self.subTest(model=source.relative_to(root), reference=reference):
                    self.assertFalse(Path(reference).is_absolute())
                    self.assertNotIn("://", reference)
                    prefix = meshdir if element.tag == "mesh" and element.get("file") else ""
                    if element.tag == "texture":
                        prefix = texturedir
                    dependency = (source.parent / prefix / reference).resolve()
                    self.assertTrue(dependency.is_relative_to(root), str(dependency))
                    self.assertTrue(dependency.is_file(), str(dependency))
                    references += 1
        self.assertGreater(references, 600)

    def test_missing_source_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "DEXTELE_ROOT"):
                assets.prepare_asset(directory, Path(directory) / "output")

    def test_single_right_chain_and_portable_meshes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = assets.prepare_asset(assets.BUNDLED_ASSET_ROOT, output_dir=directory)
            robot = ET.parse(path).getroot()
            names = [j.get("name") for j in robot.findall("joint") if j.get("type") == "revolute"]
            self.assertEqual(tuple(names), assets.DOF_NAMES)
            self.assertFalse(any("left" in link.get("name") or link.get("name").endswith("_L") for link in robot.findall("link")))
            for mesh in robot.findall(".//mesh"):
                self.assertFalse(Path(mesh.get("filename")).is_absolute())
                self.assertTrue((path.parent / mesh.get("filename")).is_file())
            for tip in assets.FINGERTIP_LINKS:
                self.assertIsNotNone(robot.find(f"link[@name='{tip}']/collision"))
            manifest = json.loads((path.parent / "manifest.json").read_text())
            self.assertEqual(len(manifest["palm_inertia_source_sha256"]), 64)
            self.assertTrue(Path(manifest["palm_inertia_source"]).is_file())
            # Proximal link2 mask 2 must not self-collide with palm mask 1.
            self.assertTrue(any(set(pair) == {"right_wuji_palm", "right_finger2_link2"}
                                for pair in manifest["contact_exclusions"]))

    def test_raised_reset_pose_clears_table_and_preserves_palm_orientation(self):
        # Canonical fingertips intersect the 0.60 m tabletop at the task mount.
        # The new reset must lift the hand without changing the base or palm XY.
        with tempfile.TemporaryDirectory() as directory:
            path = assets.prepare_asset(output_dir=directory)
            hand = dict(zip(assets.HAND_JOINT_NAMES, (0.05,) + (0.,) * 19))
            def fk(arm):
                return assets.forward_kinematics(path, {**hand, **dict(zip(assets.ARM_JOINT_NAMES, arm))})
            old, new = fk(assets.DEFAULT_ARM_POSE), fk(assets.RAISED_ARM_POSE)
            palm_old, palm_new = old[assets.PALM_LINK], new[assets.PALM_LINK]
            self.assertLess(min(old[t][2][3] + 1.01 for t in assets.FINGERTIP_LINKS), 0.60)
            self.assertGreater(min(new[t][2][3] + 1.01 for t in assets.FINGERTIP_LINKS), 0.77)
            for row in range(3):
                for col in range(3):
                    self.assertAlmostEqual(palm_old[row][col], palm_new[row][col], places=4)
            for row in (0, 1):
                self.assertAlmostEqual(palm_old[row][3], palm_new[row][3], places=4)
            self.assertAlmostEqual(palm_new[2][3] - palm_old[2][3], 0.20, places=4)
            limits = {j["name"]: j for j in json.loads((path.parent / "manifest.json").read_text())["joints"]}
            for name, value in zip(assets.ARM_JOINT_NAMES, assets.RAISED_ARM_POSE):
                self.assertLessEqual(limits[name]["lower"], value)
                self.assertLessEqual(value, limits[name]["upper"])

    def test_dextele_abduction_limits_only_change_four_generated_joints(self):
        source = assets.find_source(assets.BUNDLED_ASSET_ROOT)
        source_bytes = source.read_bytes()
        original = {
            joint.get("name"): tuple(map(float, joint.get("range").split()))
            for joint in ET.parse(source).findall(".//joint[@range]")
        }
        restricted = {f"right_finger{finger}_joint2" for finger in range(2, 6)}
        with tempfile.TemporaryDirectory() as directory:
            path = assets.prepare_asset(output_dir=directory, restrict_abduction=True)
            manifest = json.loads((path.parent / "manifest.json").read_text())
            self.assertTrue(manifest["restrict_abduction"])
            manifest_limits = {joint["name"]: (joint["lower"], joint["upper"])
                               for joint in manifest["joints"]}
            for joint in ET.parse(path).findall("joint[@type='revolute']"):
                name = joint.get("name")
                limit = joint.find("limit")
                actual = (float(limit.get("lower")), float(limit.get("upper")))
                expected = (-0.03491, 0.03491) if name in restricted else original[name]
                with self.subTest(joint=name):
                    self.assertEqual(actual, expected)
                    self.assertEqual(manifest_limits[name], expected)
        self.assertEqual(source.read_bytes(), source_bytes)

    def test_independent_source_and_urdf_fk(self):
        """Catch frame/quat conversion mistakes over the whole articulated tree."""
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation
            from yourdfpy import URDF
        except ImportError:
            self.skipTest("Independent FK check requires numpy, scipy and yourdfpy")
        source = assets.find_source(assets.BUNDLED_ASSET_ROOT)
        base = ET.parse(source).find("worldbody/body[@name='base']")
        with tempfile.TemporaryDirectory() as directory:
            robot = URDF.load(assets.prepare_asset(assets.BUNDLED_ASSET_ROOT, output_dir=directory), load_meshes=False)
            rng = np.random.default_rng(42)
            for _ in range(8):
                q = {}
                for body in base.iter("body"):
                    joint = body.find("joint")
                    if joint is not None and joint.get("name") in assets.DOF_NAMES:
                        q[joint.get("name")] = rng.uniform(*map(float, joint.get("range").split()))
                robot.update_cfg(q)

                def check(body, parent):
                    if body.get("name") == "Link1_L":
                        return
                    transform = np.eye(4)
                    if body is not base:
                        transform[:3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
                        if body.get("quat"):
                            w, x, y, z = map(float, body.get("quat").split())
                            transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
                        elif body.get("xyaxes"):
                            x, y = np.fromstring(body.get("xyaxes"), sep=" ").reshape(2, 3)
                            x = x / np.linalg.norm(x)
                            y = y - x * np.dot(x, y)
                            y = y / np.linalg.norm(y)
                            transform[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
                    joint = body.find("joint")
                    if joint is not None:
                        rotation = np.eye(4)
                        axis = np.fromstring(joint.get("axis"), sep=" ")
                        rotation[:3, :3] = Rotation.from_rotvec(axis * q[joint.get("name")]).as_matrix()
                        transform = transform @ rotation
                    world = parent @ transform
                    np.testing.assert_allclose(robot.get_transform(body.get("name")), world, atol=2e-10)
                    for child in body.findall("body"):
                        check(child, world)

                check(base, np.eye(4))


if __name__ == "__main__":
    unittest.main()
