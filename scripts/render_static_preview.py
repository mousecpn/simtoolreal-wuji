"""Render project URDF assets on the CPU, without starting Isaac Sim."""

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yourdfpy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/static-preview.png"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py"
    tree = ast.parse(source.read_text())
    arm_pose = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ARM_DEFAULT_JOINT_POS"
    )
    figure = plt.figure(figsize=(12, 9), facecolor="#eef1f3")
    axes = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axes.set_facecolor("#eef1f3")
    assets = [
        ("assets/urdf/table_narrow.urdf", (0.0, 0.0, 0.0), {}),
        (
            "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
            (0.0, 0.8, 0.0),
            arm_pose,
        ),
    ]
    mesh_count = 0
    for filename, translation, joints in assets:
        model = yourdfpy.URDF.load(str(root / filename), load_meshes=True)
        if joints:
            model.update_cfg(joints)
        for node_name in model.scene.graph.nodes_geometry:
            transform, geometry_name = model.scene.graph[node_name]
            mesh = model.scene.geometry[geometry_name]
            vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
            vertices += np.asarray(translation)
            triangles = vertices[mesh.faces]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            light = np.array([0.3, -0.5, 0.8])
            brightness = 0.4 + 0.6 * np.abs(normals @ (light / np.linalg.norm(light)))
            color = np.asarray(mesh.visual.main_color[:3], dtype=float) / 255.0
            collection = Poly3DCollection(
                triangles,
                facecolors=np.clip(brightness[:, None] * color, 0.0, 1.0),
                edgecolors="none",
                rasterized=True,
            )
            axes.add_collection3d(collection)
            mesh_count += 1
    axes.set(xlim=(-0.65, 0.65), ylim=(-0.6, 1.05), zlim=(-0.05, 1.35))
    axes.set_box_aspect((1.3, 1.65, 1.4))
    axes.view_init(elev=24, azim=-65)
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Z (m)")
    axes.set_title("KUKA iiwa + SHARPA hand | Static asset preview", pad=18)
    figure.text(
        0.5, 0.035,
        "Default joint pose / CPU mesh rendering / NOT a policy rollout / No task object",
        ha="center", fontsize=11, color="#444444",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.out} ({mesh_count} visual meshes)")


if __name__ == "__main__":
    main()