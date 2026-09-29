from pathlib import Path

from dataset_automation.images import match_poses_to_images


def test_matches_label_to_jpg_stem_case_insensitive() -> None:
    result = match_poses_to_images(
        ["IMG_0000", "IMG_0001"], [Path("a/IMG_0000.JPG"), Path("a/IMG_0001.jpg")]
    )

    assert result.matched == {
        "IMG_0000": Path("a/IMG_0000.JPG"),
        "IMG_0001": Path("a/IMG_0001.jpg"),
    }
    assert result.poses_without_image == []
    assert result.images_without_pose == []


def test_reports_poses_without_image() -> None:
    result = match_poses_to_images(["IMG_0000", "IMG_0001"], [Path("IMG_0000.jpg")])

    assert result.poses_without_image == ["IMG_0001"]


def test_reports_images_without_pose() -> None:
    result = match_poses_to_images(
        ["IMG_0000"], [Path("IMG_0000.jpg"), Path("IMG_9999.jpg")]
    )

    assert result.images_without_pose == [Path("IMG_9999.jpg")]


def test_ignores_non_jpg_files() -> None:
    result = match_poses_to_images(
        ["IMG_0000"], [Path("IMG_0000.jpg"), Path("notes.txt")]
    )

    assert result.images_without_pose == []
