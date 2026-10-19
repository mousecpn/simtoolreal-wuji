"""Render recorded simulation poses to MP4 using CPU mesh rasterization."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/simtoolreal-numba")
import cv2
import imageio_ffmpeg
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation
import yourdfpy


def pose_matrix(pose):
    pose = np.asarray(pose)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


class MeshModel:
    def __init__(self, filename):
        self.model = yourdfpy.URDF.load(str(filename), load_meshes=True)
        self.parts = []
        self.faces, self.colors = [], []
        offset = 0
        for node in self.model.scene.graph.nodes_geometry:
            _, name = self.model.scene.graph[node]
            mesh = self.model.scene.geometry[name]
            self.parts.append((node, np.asarray(mesh.vertices)))
            self.faces.append(np.asarray(mesh.faces) + offset)
            self.colors.append(np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float64))
            offset += len(mesh.vertices)
        self.faces = np.concatenate(self.faces).astype(np.int32)
        self.colors = np.concatenate(self.colors)

    def vertices(self, pose, joints=None):
        if joints:
            self.model.update_cfg(joints)
        root = pose_matrix(pose)
        parts = []
        for node, vertices in self.parts:
            transform = root @ self.model.scene.graph[node][0]
            parts.append(vertices @ transform[:3, :3].T + transform[:3, 3])
        return np.concatenate(parts)

    def framing_vertices(self, pose, joints=None, hands_only=False):
        """Sample original mesh vertices for one fixed camera covering the rollout."""
        if joints:
            self.model.update_cfg(joints)
        root = pose_matrix(pose)
        parts = []
        for node, vertices in self.parts:
            if hands_only and not any(word in str(node).lower() for word in ("finger", "palm", "wuji")):
                continue
            transform = root @ self.model.scene.graph[node][0]
            sample = vertices[::max(1, len(vertices) // 250)]
            parts.append(sample @ transform[:3, :3].T + transform[:3, 3])
        if not parts:
            return self.framing_vertices(pose, joints)
        return np.concatenate(parts)


@njit(cache=True)
def rasterize(vertices, faces, colors, image, depth, camera, center, scale):
    """Orthographic, opaque triangle rasterizer with per-pixel depth testing."""
    height, width = image.shape[:2]
    projected = np.empty((len(vertices), 3))
    for i in range(len(vertices)):
        x, y, z = vertices[i] - center
        projected[i, 0] = width / 2 + scale * (x * camera[0, 0] + y * camera[0, 1] + z * camera[0, 2])
        projected[i, 1] = height / 2 - scale * (x * camera[1, 0] + y * camera[1, 1] + z * camera[1, 2])
        projected[i, 2] = x * camera[2, 0] + y * camera[2, 1] + z * camera[2, 2]
    for f in range(len(faces)):
        i0, i1, i2 = faces[f]
        a, b, c = projected[i0], projected[i1], projected[i2]
        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(width - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(height - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        if x0 > x1 or y0 > y1:
            continue
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        u = vertices[i1] - vertices[i0]
        v = vertices[i2] - vertices[i0]
        nx, ny, nz = u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]
        normal_length = max((nx * nx + ny * ny + nz * nz) ** .5, 1e-12)
        brightness = .48 + .52 * abs((.35 * nx - .5 * ny + .8 * nz) / (1.00623 * normal_length))
        for py in range(y0, y1 + 1):
            for px in range(x0, x1 + 1):
                w0 = ((b[1] - c[1]) * (px + .5 - c[0]) + (c[0] - b[0]) * (py + .5 - c[1])) / denominator
                w1 = ((c[1] - a[1]) * (px + .5 - c[0]) + (a[0] - c[0]) * (py + .5 - c[1])) / denominator
                w2 = 1 - w0 - w1
                if min(w0, w1, w2) < -1e-7:
                    continue
                z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                if z > depth[py, px]:
                    depth[py, px] = z
                    for channel in range(3):
                        image[py, px, channel] = min(255., colors[f, channel] * brightness)


def camera_basis(elevation=22, azimuth=-62):
    el, az = np.deg2rad([elevation, azimuth])
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    return np.array([right, np.cross(forward, right), forward])


def draw_label(image, text, xy, size=.6, color=(30, 50, 70)):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def dual_camera_frames(models, frames, camera, panel_widths, panel_height):
    """Fit both arms/hands over all recorded states; keep the camera stationary."""
    bounds = [[], []]
    for frame in frames:
        for name, model in models.items():
            joints = frame.get("joint_positions") if name == "robot" else None
            vertices = model.framing_vertices(frame[f"{name}_root_pose"], joints)
            bounds[0].append(vertices @ camera.T)
            if name != "table":
                vertices = model.framing_vertices(frame[f"{name}_root_pose"], joints, hands_only=name == "robot")
                bounds[1].append(vertices @ camera.T)
    result = []
    for points, width in zip(bounds, panel_widths):
        points = np.concatenate(points)
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = camera.T @ ((lower + upper) / 2)
        extent = np.maximum(upper - lower, .01)
        scale = .88 * min(width / extent[0], panel_height / extent[1])
        result.append((center, scale))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    data = json.loads(args.trajectory.read_text())
    frames = data["frames"][:args.max_frames]
    if not frames:
        parser.error("trajectory has no frames")
    dual = data.get("arms", "single") == "dual"
    fps = data.get("fps", 30)
    model_names = [name for name in ("robot", "table", "object", "object2") if f"{name}_urdf" in data]
    models = {name: MeshModel(data[f"{name}_urdf"]) for name in model_names}
    # Distinguish the independently simulated tools even when geometry matches.
    if "object2" in models:
        models["object"].colors[:] = (225, 132, 48)
        models["object2"].colors[:] = (52, 158, 192)
    faces, colors, offset = [], [], 0
    for model in models.values():
        faces.append(model.faces + offset)
        colors.append(model.colors)
        offset += sum(len(v) for _, v in model.parts)
    faces = np.concatenate(faces)
    colors = np.concatenate(colors)
    width = args.width // 2 * 2
    height = int(width * 9 / 16) // 2 * 2
    left_width = int(width * .53)
    panel_height = height - 150
    camera = camera_basis()
    framing = (dual_camera_frames(models, frames, camera, (left_width, width - left_width), panel_height)
               if dual else [(np.array([-.16, .07, .79]), panel_height / 1.18),
                             (np.array([-.015, .075, .63]), panel_height / .66)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.out), (width, height), fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
        macro_block_size=1, ffmpeg_log_level="error",
    )
    writer.send(None)
    preview_indices = {0, len(frames) // 2, len(frames) - 1}
    try:
        for index, frame in enumerate(frames):
            vertices = np.concatenate([
                model.vertices(frame[f"{name}_root_pose"], frame.get("joint_positions") if name == "robot" else None)
                for name, model in models.items()
            ])
            image = np.full((height, width, 3), (243, 245, 247), dtype=np.uint8)
            panels = [
                (image[105:105 + panel_height, :left_width], *framing[0]),
                (image[105:105 + panel_height, left_width:], *framing[1]),
            ]
            for panel, center, scale in panels:
                depth = np.full(panel.shape[:2], -np.inf)
                rasterize(vertices, faces, colors, panel, depth, camera, center, scale)
            draw_label(image, "Tianji arms + Wuji hands | SimToolReal" if dual else "Tianji arm + Wuji hand | SimToolReal", (28, 34), .85)
            elapsed = frame["step"] * data.get("step_dt", 1 / 60)
            draw_label(image, f"Random actions | actual simulated poses | {elapsed:.2f} s", (28, 65), .57, (85, 101, 118))
            draw_label(image, "Dual arms + dual hands" if dual else "Single arm + single hand", (28, 96), .58)
            closeup_label = "Both hands + two tools" if "object2" in models else ("Both hands + task object" if dual else "Hand + task object")
            draw_label(image, closeup_label, (left_width + 20, 96), .58)
            draw_label(image, "CPU rendering of original URDF meshes | Untrained policy", (28, height - 18), .5, (85, 101, 118))
            writer.send(image)
            if index in preview_indices:
                path = args.out.with_name(f"{args.out.stem}_frame_{index:04d}.png")
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if index % 30 == 0:
                print(f"Rendered {index + 1}/{len(frames)} frames", flush=True)
    finally:
        writer.close()
    print(f"Saved {args.out} ({len(frames)} frames at {fps} fps)")


if __name__ == "__main__":
    main()
