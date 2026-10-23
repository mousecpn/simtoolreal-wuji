"""Render recorded PhysX C4 policy trajectories with the original URDF meshes."""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from render_tianji_wuji_video import (
    MeshModel,
    camera_basis,
    draw_label,
    pose_matrix,
    rasterize,
)
from scipy.spatial.transform import Rotation


def world_hand_points(model, frame):
    model.model.update_cfg(frame["joint_positions"])
    root = pose_matrix(frame["robot_root_pose"])
    parts = []
    for node, vertices in model.parts:
        if "left_" not in str(node):
            continue
        transform = root @ model.model.scene.graph[node][0]
        parts.append(
            vertices[:: max(1, len(vertices) // 120)] @ transform[:3, :3].T
            + transform[:3, 3]
        )
    return np.concatenate(parts)


def fit(points, camera, width, height, min_extent=0.01):
    projected = points @ camera.T
    lower, upper = projected.min(0), projected.max(0)
    center = camera.T @ ((lower + upper) / 2)
    extent = np.maximum(upper - lower, min_extent)
    return center, 0.86 * min(width / extent[0], height / extent[1])


def goal_box(panel, frame, dimensions, camera, center, scale):
    q = np.array(frame["goal_root_pose"])
    offsets = np.array(list(itertools.product((-1, 1), repeat=3))) * dimensions / 2
    points = offsets @ Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix().T + q[:3]
    projected = (points - center) @ camera.T
    xy = np.stack(
        (
            panel.shape[1] / 2 + scale * projected[:, 0],
            panel.shape[0] / 2 - scale * projected[:, 1],
        ),
        -1,
    )
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j) in (1, 2, 4):
                cv2.line(
                    panel,
                    tuple(xy[i].astype(int)),
                    tuple(xy[j].astype(int)),
                    (24, 165, 183),
                    1,
                    cv2.LINE_AA,
                )


def render(path, width=1280, max_frames=None):
    data = json.loads(path.read_text())
    frames = data["frames"][:max_frames]
    models = {
        name: MeshModel(data[f"{name}_urdf"]) for name in ("robot", "table", "object")
    }
    models["object"].colors[:] = (222, 142, 54)
    faces, colors, offset = [], [], 0
    for model in models.values():
        faces.append(model.faces + offset)
        colors.append(model.colors)
        offset += sum(len(v) for _, v in model.parts)
    faces = np.concatenate(faces)
    colors = np.concatenate(colors)
    height = int(width * 9 / 16) // 2 * 2
    header = 105
    footer = 32
    panel_height = height - header - footer
    left_width = width // 2
    camera = camera_basis(elevation=24, azimuth=-65)
    initial = frames[0]
    full = np.concatenate(
        [
            models["robot"].framing_vertices(f["robot_root_pose"], f["joint_positions"])
            for f in frames[:: max(1, len(frames) // 50)]
        ]
        + [
            models[n].framing_vertices(initial[f"{n}_root_pose"])
            for n in ("table", "object")
        ]
    )
    hands = np.concatenate(
        [
            world_hand_points(models["robot"], f)
            for f in frames[:: max(1, len(frames) // 50)]
        ]
    )
    close = np.concatenate(
        (hands, models["object"].framing_vertices(initial["object_root_pose"]))
    )
    framing = [
        fit(full, camera, left_width, panel_height),
        fit(close, camera, width - left_width, panel_height, 0.25),
    ]
    out = path.with_suffix(".mp4")
    writer = imageio_ffmpeg.write_frames(
        str(out),
        (width, height),
        fps=data["fps"],
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    print(
        f"[render] {path.stem}: {len(frames)} frames, {len(faces)} mesh triangles",
        flush=True,
    )
    try:
        for index, frame in enumerate(frames):
            vertices = np.concatenate(
                [
                    m.vertices(
                        frame[f"{n}_root_pose"],
                        frame["joint_positions"] if n == "robot" else None,
                    )
                    for n, m in models.items()
                ]
            )
            image = np.full((height, width, 3), (243, 245, 247), dtype=np.uint8)
            for panel, (center, scale) in zip(
                (
                    image[header : header + panel_height, :left_width],
                    image[header : header + panel_height, left_width:],
                ),
                framing,
            ):
                depth = np.full(panel.shape[:2], -np.inf)
                rasterize(vertices, faces, colors, panel, depth, camera, center, scale)
                goal_box(
                    panel,
                    frame,
                    np.array(data["box_dimensions"]),
                    camera,
                    center,
                    scale,
                )
            draw_label(image, data["title"], (24, 30), 0.72)
            elapsed = frame["step"] * data["step_dt"]
            draw_label(
                image,
                f"{data['tool'].upper()} | {data['action_mode']} | t = {elapsed:.2f}s | seed {data['seed']}",
                (24, 57),
                0.55,
            )
            draw_label(image, "Whole arm", (24, 87), 0.52)
            draw_label(
                image, "Hand + object | cyan: goal box", (left_width + 18, 87), 0.52
            )
            cv2.line(
                image,
                (left_width, header),
                (left_width, height - footer),
                (215, 222, 227),
                1,
            )
            draw_label(
                image,
                "1 PPO epoch (smoke checkpoint) | Actual PhysX poses, CPU mesh rendering",
                (24, height - 12),
                0.48,
                (85, 101, 118),
            )
            writer.send(image)
            if index in (0, len(frames) // 2, len(frames) - 1):
                cv2.imwrite(
                    str(out.with_name(f"{out.stem}_frame_{index:04d}.png")),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
            if index % 60 == 0:
                print(f"[render] {path.stem}: {index + 1}/{len(frames)}", flush=True)
    finally:
        writer.close()
    print(f"[render] SAVED {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    for name in ("hammer", "screwdriver", "marker"):
        render(args.directory / f"{name}.json", args.width, args.max_frames)


if __name__ == "__main__":
    main()
