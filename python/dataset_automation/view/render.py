"""Draw a scene with Open3D: an interactive window, or an image rendered off screen.

The only module that imports Open3D, an optional dependency (`uv sync --extra view`).
"""

from collections.abc import Sequence

import numpy as np
import open3d as o3d  # pyright: ignore[reportMissingImports]

from dataset_automation.view.scene import LineSet

BACKGROUND_RGBA = (1.0, 1.0, 1.0, 1.0)
FIELD_OF_VIEW_DEG = 50.0
LINE_WIDTH_PX = 1.5


def to_open3d(lines: LineSet):
    """An `open3d.geometry.LineSet` (no return annotation: Open3D ships no type stubs)."""
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(lines.points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines.segments.astype(np.int32))
    line_set.colors = o3d.utility.Vector3dVector(lines.colours.astype(np.float64))
    return line_set


def show(layers: Sequence[LineSet], title: str) -> None:
    """Open a window to turn around the scene; returns when it is closed."""
    o3d.visualization.draw_geometries(
        [to_open3d(layer) for layer in layers if len(layer)], window_name=title
    )


def render_image(
    layers: Sequence[LineSet], frame_on: np.ndarray, width_px: int, height_px: int
) -> np.ndarray:
    """RGB image (height, width, 3) of the scene seen from above and to the south, tilted
    enough that camera heights read as relief, framing the points `frame_on` (P, 3)."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width_px, height_px)
    renderer.scene.set_background(np.array(BACKGROUND_RGBA))
    # Tone mapping would wash out the colour scale of the links and tint the background.
    renderer.scene.view.set_post_processing(False)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "unlitLine"
    material.line_width = LINE_WIDTH_PX

    drawn = [layer for layer in layers if len(layer)]
    for number, layer in enumerate(drawn):
        renderer.scene.add_geometry(f"layer_{number}", to_open3d(layer), material)

    centre, eye = viewpoint(frame_on)
    renderer.setup_camera(FIELD_OF_VIEW_DEG, centre, eye, np.array([0.0, 0.0, 1.0]))
    return np.asarray(renderer.render_to_image())


def viewpoint(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point looked at and eye position that frame `points` (P, 3), z up."""
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    extent = float(np.linalg.norm(high - low))
    return centre, centre + extent * np.array([0.0, -0.75, 0.75])
