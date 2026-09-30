"""Download the Mermaid dataset (SEANOE, doi:10.17882/97987) into data/mermaid/.

    uv run python scripts/fetch_mermaid.py            # poses, GCP network, reports (~5 MB)
    uv run python scripts/fetch_mermaid.py --images   # + the 1,244 images (3.69 GB zip, extracted)

Files already present with the expected size are skipped, and an interrupted download resumes
where it stopped. Licence CC BY-NC-ND: never commit them (data/ is gitignored, and the
repository is public).
"""

import argparse
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
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

# SEANOE throttles each connection (~0.3 MB/s measured) and stalls on large single requests,
# so files are fetched as parallel byte ranges: 8 connections measured at ~1.6 MB/s.
CHUNK_BYTES = 4 << 20
PARALLEL_REQUESTS = 8
REQUEST_TIMEOUT_S = 60
MAX_ATTEMPTS = 5


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

    # Written under a temporary name so an interrupted download is never mistaken for a complete
    # one; `.done` lists the chunks already on disk so a rerun only fetches the missing ones.
    partial = target.with_name(target.name + ".part")
    progress = target.with_name(target.name + ".part.done")
    chunk_count = -(-expected_size // CHUNK_BYTES)
    done = _completed_chunks(progress)
    missing = [i for i in range(chunk_count) if i not in done]
    print(
        f"downloading {name} ({expected_size / 1e6:.1f} MB, {len(missing)}/{chunk_count} chunks to fetch)"
    )

    with partial.open("ab") as output:
        output.truncate(expected_size)
    descriptor = os.open(partial, os.O_WRONLY)
    try:
        _fetch_chunks(
            url,
            expected_size,
            missing,
            descriptor,
            progress,
            chunk_count - len(missing),
        )
    finally:
        os.close(descriptor)

    partial.rename(target)
    progress.unlink(missing_ok=True)
    return target


def _fetch_chunks(
    url: str,
    size: int,
    chunks: list[int],
    descriptor: int,
    progress: Path,
    already_done: int,
) -> None:
    total = already_done + len(chunks)
    with ThreadPoolExecutor(PARALLEL_REQUESTS) as pool, progress.open("a") as log:
        futures = {
            pool.submit(
                _fetch_range, url, i * CHUNK_BYTES, min(size, (i + 1) * CHUNK_BYTES)
            ): i
            for i in chunks
        }
        for finished, future in enumerate(
            as_completed(futures), start=already_done + 1
        ):
            chunk = futures[future]
            # pwrite takes an explicit offset, so parallel chunks never race on a file position.
            os.pwrite(descriptor, future.result(), chunk * CHUNK_BYTES)
            log.write(f"{chunk}\n")
            log.flush()
            if finished % 25 == 0 or finished == total:
                print(
                    f"  {finished}/{total} chunks ({100 * finished / total:.0f} %)",
                    flush=True,
                )


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                data = response.read()
            if len(data) == end - start:
                return data
        except (urllib.error.URLError, HTTPException, TimeoutError, OSError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError(
        f"bytes {start}-{end - 1} of {url}: failed after {MAX_ATTEMPTS} attempts"
    )


def _completed_chunks(progress: Path) -> set[int]:
    if not progress.exists():
        return set()
    return {int(line) for line in progress.read_text().split()}


def extract_images(archive: Path) -> None:
    if IMAGES_DIRECTORY.exists():
        print(f"up to date  {IMAGES_DIRECTORY}/")
        return
    print(f"extracting  {archive.name} into {IMAGES_DIRECTORY}/")
    partial = IMAGES_DIRECTORY.with_name(IMAGES_DIRECTORY.name + ".part")
    with zipfile.ZipFile(archive) as zipped:
        # The archive was made on macOS: skip its AppleDouble metadata folder.
        members = [
            name for name in zipped.namelist() if not name.startswith("__MACOSX/")
        ]
        zipped.extractall(partial, members=members)
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
