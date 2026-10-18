"""Run without importing Isaac Sim: python -m unittest discover -s tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


MODULE = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/tianji_wuji/assets.py"
SPEC = importlib.util.spec_from_file_location("tianji_wuji_assets", MODULE)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def test_missing_source_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "DEXTELE_ROOT"):
                assets.prepare_asset(directory, Path(directory) / "output")

    def test_single_right_chain_and_portable_meshes(self):
        try:
            assets.find_source()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            path = assets.prepare_asset(output_dir=directory)
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

    def test_independent_source_and_urdf_fk(self):
        """Catch frame/quat conversion mistakes over the whole articulated tree."""
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation
            from yourdfpy import URDF
        except ImportError:
            self.skipTest("Independent FK check requires numpy, scipy and yourdfpy")
        try:
            source = assets.find_source()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        base = ET.parse(source).find("worldbody/body[@name='base']")
        with tempfile.TemporaryDirectory() as directory:
            robot = URDF.load(assets.prepare_asset(output_dir=directory), load_meshes=False)
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
