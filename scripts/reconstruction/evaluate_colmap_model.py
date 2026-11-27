"""Compare the cameras COLMAP placed with the reference poses of Mermaid.

    uv run python scripts/reconstruction/evaluate_colmap_model.py \
        outputs/colmap/all_images/sparse_text/images.txt \
        --image-list outputs/colmap/all_images/image_list.txt

`images.txt` is a COLMAP model exported as text (`colmap model_converter --output_type TXT`).
The reconstruction has its own origin, orientation and scale: it is first brought onto the
reference by the best similarity between the camera centres, and what is reported is the
distance left between each camera and its reference position.

The reference poses come from Metashape and are themselves a reconstruction: a small distance
says the two agree, not that either is exact.
"""

import argparse
from pathlib import Path

import numpy as np

from dataset_automation.reconstruction.alignment import fit_similarity
from dataset_automation.reconstruction.colmap_files import read_camera_centres
from dataset_automation.reference.mermaid import load_mermaid_reference

POSES_PATH = Path("data/mermaid/107177.xml")


def main() -> None:
    arguments = parse_arguments()
    reference = load_mermaid_reference(arguments.mermaid_poses)
    reference_centre = dict(zip(reference.poses.labels, reference.centres, strict=True))
    placed = read_camera_centres(arguments.images_file)
    known = {
        name: centre
        for name, centre in placed.items()
        if stem(name) in reference_centre
    }

    if arguments.image_list is not None:
        asked = [line for line in arguments.image_list.read_text().splitlines() if line]
        print(
            f"images placed: {len(placed)} of {len(asked)} ({len(placed) / len(asked):.1%})"
        )
    else:
        print(f"images placed: {len(placed)}")
    print(f"of which with a reference pose: {len(known)}")
    if len(known) < 3:
        raise SystemExit("too few cameras with a reference pose to compare positions")

    centres = np.array(list(known.values()))
    expected = np.array([reference_centre[stem(name)] for name in known])
    similarity = fit_similarity(centres, expected)
    distance = np.linalg.norm(similarity.apply(centres) - expected, axis=1)
    print(f"scale of the reconstruction: 1 unit = {similarity.scale:.4f} m")
    print(
        "distance to the reference position: "
        f"median {100 * np.median(distance):.1f} cm, "
        f"nine in ten below {100 * np.percentile(distance, 90):.1f} cm, "
        f"worst {100 * distance.max():.1f} cm"
    )
    for limit_m in (0.05, 0.20, 1.0):
        print(
            f"  cameras beyond {100 * limit_m:.0f} cm: {int((distance > limit_m).sum())}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images_file", type=Path)
    parser.add_argument(
        "--image-list", type=Path, help="the images COLMAP was asked to reconstruct"
    )
    parser.add_argument("--mermaid-poses", type=Path, default=POSES_PATH)
    return parser.parse_args()


def stem(image_name: str) -> str:
    return Path(image_name).stem


if __name__ == "__main__":
    main()
