"""Download the Mermaid dataset (SEANOE, doi:10.17882/97987) into data/mermaid/.

    uv run python scripts/fetch_mermaid.py            # poses, GCP network, reports (~5 MB)
    uv run python scripts/fetch_mermaid.py --images   # + the 1,244 images (3.69 GB zip, extracted)

Files already present with the expected size are skipped. Licence CC BY-NC-ND: never commit
them (data/ is gitignored, and the repository is public).
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www.seanoe.org/data/00868/97987/data"
DESTINATION = Path("data/mermaid")
METADATA_FILES = {
    "107177.xml": "camera poses from the bundle adjustment",
    "107176.stl": "micro geodesic network (5 GCPs)",
    "107174.pdf": "image data report (calibration, GSD)",
    "107175.pdf": "micro geodesic network report",
}
IMAGES_ARCHIVE = "107179.zip"
IMAGES_DIRECTORY = DESTINATION / "images"


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def download(name: str) -> Path:
    url = f"{BASE_URL}/{name}"
    target = DESTINATION / name
    expected_size = remote_size(url)
    if target.exists() and target.stat().st_size == expected_size:
        print(f"up to date  {name}")
        return target

    # Written under a temporary name so an interrupted download is never mistaken for a complete one.
    partial = target.with_suffix(target.suffix + ".part")
    print(f"downloading {name} ({expected_size / 1e6:.1f} MB)")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1 << 20)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"{name}: got {partial.stat().st_size} bytes, expected {expected_size}"
        )
    partial.rename(target)
    return target


def extract_images(archive: Path) -> None:
    if IMAGES_DIRECTORY.exists():
        print(f"up to date  {IMAGES_DIRECTORY}/")
        return
    print(f"extracting  {archive.name} into {IMAGES_DIRECTORY}/")
    partial = IMAGES_DIRECTORY.with_name(IMAGES_DIRECTORY.name + ".part")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(partial)
    partial.rename(IMAGES_DIRECTORY)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--images", action="store_true", help="also fetch the 3.69 GB image archive"
    )
    arguments = parser.parse_args()

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        download(name)
    if arguments.images:
        extract_images(download(IMAGES_ARCHIVE))


if __name__ == "__main__":
    main()
