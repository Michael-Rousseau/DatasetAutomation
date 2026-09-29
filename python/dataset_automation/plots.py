import numpy as np
from matplotlib.figure import Figure

from dataset_automation.poses import CameraPoses, camera_centres, optical_axes

# Arrows show the horizontal part of the viewing direction; for nadir views they are short.
ARROW_LENGTH_M = 0.5


def plot_camera_top_view(poses: CameraPoses) -> Figure:
    centres = camera_centres(poses)
    axes_xy = optical_axes(poses)[:, :2] * ARROW_LENGTH_M
    capture_order = np.arange(len(poses))

    figure = Figure(figsize=(10, 8))
    ax = figure.subplots()
    scatter = ax.scatter(
        centres[:, 0], centres[:, 1], c=capture_order, cmap="viridis", s=8
    )
    ax.quiver(
        centres[:, 0],
        centres[:, 1],
        axes_xy[:, 0],
        axes_xy[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002,
        alpha=0.5,
    )
    figure.colorbar(scatter, ax=ax, label="capture order (image index)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Camera centres, top view ({len(poses)} cameras)")
    return figure


def plot_optical_axis_z(poses: CameraPoses) -> Figure:
    figure = Figure(figsize=(7, 4))
    ax = figure.subplots()
    ax.hist(optical_axes(poses)[:, 2], bins=60, range=(-1.0, 1.0))
    ax.set_xlabel("z component of the optical axis (−1 = looking straight down)")
    ax.set_ylabel("cameras")
    ax.set_title("Viewing direction: vertical component")
    return figure
