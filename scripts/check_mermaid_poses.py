"""Sanity check of the Mermaid poses: which direction does `transform` map?

Prints trajectory statistics under both hypotheses and saves the top-view plots.
"""

from pathlib import Path

import numpy as np

from dataset_automation.plots import plot_camera_top_view, plot_optical_axis_z
from dataset_automation.poses import CameraPoses, load_camera_poses

POSES_PATH = Path("data/mermaid/107177.xml")
OUTPUT_DIR = Path("outputs")


def invert_rigid(transforms: np.ndarray) -> np.ndarray:
    # Inverse of [R t; 0 1] is [Rᵀ −Rᵀt; 0 1]: cheaper and more exact than np.linalg.inv.
    rotations_t = transforms[:, :3, :3].transpose(0, 2, 1)
    inverse = np.zeros_like(transforms)
    inverse[:, :3, :3] = rotations_t
    inverse[:, :3, 3] = -np.einsum("nij,nj->ni", rotations_t, transforms[:, :3, 3])
    inverse[:, 3, 3] = 1.0
    return inverse


def trajectory_statistics(camera_to_world: np.ndarray) -> dict[str, float]:
    centres = camera_to_world[:, :3, 3]
    axes_z = camera_to_world[:, 2, 2]
    extent = centres.max(axis=0) - centres.min(axis=0)
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    return {
        "extent x (m)": extent[0],
        "extent y (m)": extent[1],
        "extent z (m)": extent[2],
        "median step (m)": np.median(steps),
        "95th pct step (m)": np.percentile(steps, 95),
        "max step (m)": steps.max(),
        "axes looking down (%)": 100 * np.mean(axes_z < 0),
    }


def print_comparison(poses: CameraPoses) -> None:
    hypotheses = {
        "camera->world": trajectory_statistics(poses.camera_to_world),
        "world->camera": trajectory_statistics(invert_rigid(poses.camera_to_world)),
    }
    print(f"{'':24}" + "".join(f"{name:>16}" for name in hypotheses))
    for key in hypotheses["camera->world"]:
        print(
            f"{key:24}"
            + "".join(f"{stats[key]:>16.2f}" for stats in hypotheses.values())
        )


def main() -> None:
    poses = load_camera_poses(POSES_PATH)
    print_comparison(poses)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_camera_top_view(poses).savefig(
        OUTPUT_DIR / "step1_camera_top_view.png", dpi=150
    )
    plot_optical_axis_z(poses).savefig(OUTPUT_DIR / "step1_optical_axis_z.png", dpi=150)
    print(f"figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
