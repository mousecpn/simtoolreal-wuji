"""Render a saved Tianji/Wuji rollout state with actual URDF visual meshes.

This CPU visualization reproduces the recorded poses; it is not an Isaac camera
image. No simulator imports are needed.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/simtoolreal-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import yourdfpy


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def load_triangles(filename, root_pose, joints=None, subdivide=False):
    model = yourdfpy.URDF.load(str(filename), load_meshes=True)
    if joints:
        model.update_cfg(joints)
    root = pose_matrix(root_pose)
    triangles, colors = [], []
    for node in model.scene.graph.nodes_geometry:
        local, name = model.scene.graph[node]
        mesh = model.scene.geometry[name]
        # Small table faces avoid painter-sort artifacts with the tiny tool
        # immediately above the tabletop.
        if subdivide:
            material_color = mesh.visual.face_colors[0].copy()
            for _ in range(5):
                mesh = mesh.subdivide()
            mesh.visual.face_colors = material_color
        transform = root @ local
        vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        faces = vertices[mesh.faces]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([0.35, -0.5, 0.8])
        light /= np.linalg.norm(light)
        brightness = 0.48 + 0.52 * np.abs(normals @ light)
        base = np.asarray(mesh.visual.face_colors[:, :3], dtype=float) / 255.0
        triangles.append(faces)
        colors.append(np.clip(brightness[:, None] * base, 0.0, 1.0))
    return np.concatenate(triangles), np.concatenate(colors)


def setup_axes(ax, center, extent, elevation, azimuth):
    ax.set_facecolor("#f3f5f7")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.94, 0.95, 0.97, 1))
        axis._axinfo["grid"].update(color=(0.72, 0.75, 0.8, 0.32))
    ax.set_xlim(center[0] - extent[0] / 2, center[0] + extent[0] / 2)
    ax.set_ylim(center[1] - extent[1] / 2, center[1] + extent[1] / 2)
    ax.set_zlim(center[2] - extent[2] / 2, center[2] + extent[2] / 2)
    ax.set_box_aspect(extent)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.tick_params(labelsize=8, colors="#677382")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.snapshot.read_text())
    repo = Path(__file__).resolve().parents[1]
    all_triangles, all_colors = [], []
    for name in ("robot", "table", "object"):
        path = Path(state[f"{name}_urdf"])
        if not path.is_absolute():
            path = repo / path
        triangles, colors = load_triangles(
            path, state[f"{name}_root_pose"],
            state.get("joint_positions") if name == "robot" else None,
            subdivide=name == "table",
        )
        all_triangles.append(triangles)
        all_colors.append(colors)
    triangles = np.concatenate(all_triangles)
    colors = np.concatenate(all_colors)
    vertices = triangles.reshape(-1, 3)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    extent = np.maximum((upper - lower) * 1.12, 0.4)
    # Both views use one collection so depth sorting also works across links.
    fig = plt.figure(figsize=(16, 8), facecolor="#f3f5f7")
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    overview = fig.add_subplot(121, projection="3d")
    detail = fig.add_subplot(122, projection="3d")
    setup_axes(overview, center, extent, 22, -62)
    object_center = np.asarray(state["object_root_pose"][:3])
    detail_center = object_center + np.array([0., 0., 0.08])
    setup_axes(detail, detail_center, np.array([0.65, 0.65, 0.65]), 24, -65)
    for ax in (overview, detail):
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolors=colors, edgecolors="none", zsort="average",
            rasterized=True,
        ))
        if "goal_root_pose" in state:
            goal = state["goal_root_pose"][:3]
            ax.scatter(*goal, color="#00a3a3", marker="x", s=45, linewidth=1.5)
    overview.set_title("Single Tianji arm + Wuji hand", loc="left", pad=16,
                       fontsize=15, color="#253447")
    detail.set_title("Hand and task object", loc="left", pad=16,
                     fontsize=15, color="#253447")
    fig.suptitle(f"SimToolReal environment  |  rollout step {state['step']}",
                 x=0.06, ha="left", y=0.98, fontsize=21, color="#162b40")
    fig.text(0.06, 0.925, "Random actions · Untrained policy · Actual simulated joint/object poses",
             fontsize=11, color="#5c6b7c")
    fig.text(0.06, 0.04,
             "CPU rendering of URDF visual meshes  |  Cyan ×: target object position",
             fontsize=10, color="#5c6b7c")
    fig.subplots_adjust(left=0.01, right=0.98, top=0.83, bottom=0.12, wspace=0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {args.out} ({len(triangles):,} mesh triangles)")


if __name__ == "__main__":
    main()
