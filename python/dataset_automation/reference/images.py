from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PoseImageMatch:
    matched: dict[str, Path]
    poses_without_image: list[str]
    images_without_pose: list[Path]


def match_poses_to_images(labels: list[str], image_paths: list[Path]) -> PoseImageMatch:
    """Pair each pose label with the image whose stem equals it (`.jpg` extension, any case).

    Paths that are not `.jpg` files are ignored rather than reported.
    """
    images_by_stem = {
        path.stem: path for path in image_paths if path.suffix.lower() == ".jpg"
    }
    matched = {
        label: images_by_stem[label] for label in labels if label in images_by_stem
    }
    return PoseImageMatch(
        matched=matched,
        poses_without_image=[label for label in labels if label not in matched],
        images_without_pose=[
            path for stem, path in images_by_stem.items() if stem not in matched
        ],
    )
