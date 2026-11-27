"""Picture of a COLMAP reconstruction of Mermaid: its points and cameras seen from above.

    uv run python scripts/reconstruction/preview_colmap_model.py \\
        outputs/colmap/min_views_5/sparse_text/0

The folder is a model exported as text (`images.txt`, `points3D.txt`). The reconstruction is
first brought onto the reference poses, so that both panels are in metres: on the left the 3D
points in their own colours with the cameras placed; on the right the height of the points.
Written next to the model as `preview.png`.
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from dataset_automation.reconstruction.alignment import fit_similarity
from dataset_automation.reconstruction.colmap_files import (
    read_camera_centres,
    read_points,
)
from dataset_automation.reference.mermaid import load_mermaid_reference

POSES_PATH = Path("data/mermaid/107177.xml")
# Points far from the others are triangulation mistakes: they would set the colour scale.
HEIGHT_PERCENTILES = (2, 98)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--mermaid-poses", type=Path, default=POSES_PATH)
    parser.add_argument("--title", default="")
    arguments = parser.parse_args()

    reference = load_mermaid_reference(arguments.mermaid_poses)
    reference_centre = dict(zip(reference.poses.labels, reference.centres, strict=True))
    placed = read_camera_centres(arguments.model / "images.txt")
    names = [name for name in sorted(placed) if Path(name).stem in reference_centre]
    cameras = np.array([placed[name] for name in names])
    similarity = fit_similarity(
        cameras, np.array([reference_centre[Path(name).stem] for name in names])
    )
    positions, colours = read_points(arguments.model / "points3D.txt")
    points, cameras = similarity.apply(positions), similarity.apply(cameras)

    low, high = np.percentile(points[:, 2], HEIGHT_PERCENTILES)
    shown = (points[:, 2] >= low) & (points[:, 2] <= high)
    figure = Figure(figsize=(15, 7))
    left, right = figure.subplots(1, 2)
    left.scatter(
        points[shown, 0], points[shown, 1], c=colours[shown] / 255, s=0.2, linewidths=0
    )
    left.plot(cameras[:, 0], cameras[:, 1], color="red", linewidth=0.5, alpha=0.7)
    left.scatter(cameras[:, 0], cameras[:, 1], color="red", s=3)
    left.set_title(f"{len(points):,} points, {len(cameras)} cameras (red)")
    heights = right.scatter(
        points[shown, 0],
        points[shown, 1],
        c=points[shown, 2],
        s=0.2,
        linewidths=0,
        cmap="viridis",
    )
    figure.colorbar(heights, ax=right, label="height (m)")
    right.set_title("height of the points")
    for axis in (left, right):
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    if arguments.title:
        figure.suptitle(arguments.title)
    figure.tight_layout()
    output = arguments.model / "preview.png"
    figure.savefig(output, dpi=130)
    print(f"written to {output}")


if __name__ == "__main__":
    main()
