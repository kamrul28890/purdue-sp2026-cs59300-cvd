#!/usr/bin/env python
"""Collect project data used for CoTracker3 reproduction and analysis."""

from __future__ import annotations

import csv
import shutil
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA = ROOT / "project_data"
RAW_VIDEOS = PROJECT_DATA / "raw_videos"
OFFICIAL_ASSETS_DIR = RAW_VIDEOS / "official_assets"
INTERNET_ARCHIVE_DIR = RAW_VIDEOS / "internet_archive"
KUBRIC_CACHE_DIR = PROJECT_DATA / "kubric_cache"
KUBRIC_SAMPLE_DIR = RAW_VIDEOS / "kubric_sample"
MANIFESTS_DIR = PROJECT_DATA / "manifests"
DATA_INVENTORY = MANIFESTS_DIR / "data_inventory.csv"

OFFICIAL_ASSETS = [
    {
        "item_id": "asset_apple_video",
        "title": "Official apple demo video",
        "filename": "apple.mp4",
        "source_url": "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4",
        "license_or_terms": "Official repository asset",
        "notes": "Smoke-test video copied from the CoTracker repository assets folder.",
    },
    {
        "item_id": "asset_apple_mask",
        "title": "Official apple demo mask",
        "filename": "apple_mask.png",
        "source_url": "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple_mask.png",
        "license_or_terms": "Official repository asset",
        "notes": "Reference mask image that accompanies the apple demo asset.",
    },
    {
        "item_id": "asset_bmx_bumps_gif",
        "title": "Official BMX bumps demo GIF",
        "filename": "bmx-bumps.gif",
        "source_url": "https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/bmx-bumps.gif",
        "license_or_terms": "Official repository asset",
        "notes": "Animated demo clip copied from the CoTracker repository assets folder.",
    },
]

INTERNET_ARCHIVE_VIDEOS = [
    {
        "item_id": "ia_design_for_dreaming",
        "identifier": "Designfo1956",
        "title": "Design for Dreaming",
        "filename": "Designfo1956_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_willie_shoemaker",
        "identifier": "WillieSh1950",
        "title": "[Willie Shoemaker at Golden Gate Fields]",
        "filename": "WillieSh1950_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_tip_tops_in_peppyland",
        "identifier": "TipTopsi1934",
        "title": "Tip-Tops in Peppyland, The",
        "filename": "TipTopsi1934_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_trading_centers",
        "identifier": "TradingC1947",
        "title": "Trading Centers of the Pacific Coast",
        "filename": "TradingC1947_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_street_of_memory",
        "identifier": "Streetof1937",
        "title": "Street of Memory",
        "filename": "Streetof1937_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_aurora_drag_race",
        "identifier": "aurora_drag_race_set",
        "title": "Aurora Stunt and Drag Race Set Commercial",
        "filename": "aurora_drag_race_set_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_park_conscious",
        "identifier": "ParkCons1938",
        "title": "Park Conscious",
        "filename": "ParkCons1938_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_case_of_spring_fever",
        "identifier": "CaseofSp1940",
        "title": "Case of Spring Fever, A",
        "filename": "CaseofSp1940_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_122_eyes",
        "identifier": "122Eyes1950",
        "title": "122 Eyes",
        "filename": "122Eyes1950_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
    {
        "item_id": "ia_sleep_for_health",
        "identifier": "Sleepfor1950",
        "title": "Sleep for Health",
        "filename": "Sleepfor1950_512kb.mp4",
        "license_or_terms": "http://creativecommons.org/licenses/publicdomain/",
    },
]

TAPVID_DAVIS_ROW = {
    "item_id": "tapvid_davis_benchmark",
    "source_type": "benchmark",
    "title": "TAP-Vid DAVIS benchmark",
    "source_url": "https://github.com/google-deepmind/tapnet",
    "license_or_terms": "See datasets/tapvid_davis/SOURCES.md",
    "local_path": "datasets/tapvid_davis/tapvid_davis.pkl",
    "notes": "Benchmark file already downloaded for online and offline DAVIS evaluation.",
}

KUBRIC_README_ROW = {
    "item_id": "cotracker3_kubric_readme",
    "source_type": "cotracker_dataset",
    "title": "CoTracker3_Kubric dataset card",
    "source_url": "https://huggingface.co/datasets/facebook/CoTracker3_Kubric",
    "license_or_terms": "Apache-2.0",
    "local_path": "project_data/kubric_cache/CoTracker3_Kubric_README.md",
    "notes": "Dataset card copied locally for reproducibility and citation notes.",
}

KUBRIC_SHARD_ROW = {
    "item_id": "cotracker3_kubric_shard_0000",
    "source_type": "cotracker_dataset",
    "title": "CoTracker3_Kubric shard 0000",
    "source_url": "https://huggingface.co/datasets/facebook/CoTracker3_Kubric",
    "license_or_terms": "Apache-2.0",
    "local_path": "project_data/kubric_cache/0000.tar.gz",
    "notes": (
        "One official dataset shard downloaded and extracted to "
        "project_data/raw_videos/kubric_sample/0000. The shard contains annotations, "
        "trajectories, depths, and metadata rather than a ready-made MP4 clip."
    ),
}


def ensure_dirs() -> None:
    for path in [
        OFFICIAL_ASSETS_DIR,
        INTERNET_ARCHIVE_DIR,
        KUBRIC_CACHE_DIR,
        KUBRIC_SAMPLE_DIR,
        MANIFESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def request_with_user_agent(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})


def copy_if_needed(source: Path, destination: Path) -> str:
    if destination.exists():
        return "already_present"
    shutil.copy2(source, destination)
    return "copied"


def download_if_needed(url: str, destination: Path) -> str:
    if destination.exists() and destination.stat().st_size > 0:
        return "already_present"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    with urllib.request.urlopen(request_with_user_agent(url)) as response:
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    temp_path.replace(destination)
    return "downloaded"


def is_within_directory(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return False
    return True


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            member_path = destination / member.name
            if not is_within_directory(destination, member_path):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        archive.extractall(destination)


def extract_kubric_shard_if_needed(tar_path: Path, destination: Path) -> str:
    shard_dir = destination / "0000"
    if shard_dir.exists() and any(shard_dir.iterdir()):
        return "already_extracted"
    safe_extract_tar(tar_path, destination)
    return "extracted"


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def file_size_or_blank(path: Path) -> str:
    if path.exists() and path.is_file():
        return str(path.stat().st_size)
    return ""


def build_manifest_rows(
    official_statuses: dict[str, str],
    kubric_readme_status: str,
    kubric_shard_status: str,
    kubric_extract_status: str,
    internet_archive_statuses: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for asset in OFFICIAL_ASSETS:
        local_path = OFFICIAL_ASSETS_DIR / asset["filename"]
        rows.append(
            {
                "item_id": asset["item_id"],
                "source_type": "repo_asset",
                "title": asset["title"],
                "source_url": asset["source_url"],
                "license_or_terms": asset["license_or_terms"],
                "local_path": relative_path(local_path),
                "status": official_statuses[asset["filename"]],
                "size_bytes": file_size_or_blank(local_path),
                "notes": asset["notes"],
            }
        )

    davis_path = ROOT / TAPVID_DAVIS_ROW["local_path"]
    rows.append(
        {
            **TAPVID_DAVIS_ROW,
            "status": "verified_present" if davis_path.exists() else "missing",
            "size_bytes": file_size_or_blank(davis_path),
        }
    )

    kubric_readme_path = ROOT / KUBRIC_README_ROW["local_path"]
    rows.append(
        {
            **KUBRIC_README_ROW,
            "status": kubric_readme_status,
            "size_bytes": file_size_or_blank(kubric_readme_path),
        }
    )

    kubric_shard_path = ROOT / KUBRIC_SHARD_ROW["local_path"]
    rows.append(
        {
            **KUBRIC_SHARD_ROW,
            "status": kubric_shard_status,
            "size_bytes": file_size_or_blank(kubric_shard_path),
            "notes": (
                f"{KUBRIC_SHARD_ROW['notes']} Extraction status: {kubric_extract_status}."
            ),
        }
    )

    for video in INTERNET_ARCHIVE_VIDEOS:
        local_path = INTERNET_ARCHIVE_DIR / video["filename"]
        rows.append(
            {
                "item_id": video["item_id"],
                "source_type": "internet_archive",
                "title": video["title"],
                "source_url": f"https://archive.org/details/{video['identifier']}",
                "license_or_terms": video["license_or_terms"],
                "local_path": relative_path(local_path),
                "status": internet_archive_statuses[video["filename"]],
                "size_bytes": file_size_or_blank(local_path),
                "notes": f"Internet Archive identifier: {video['identifier']}.",
            }
        )

    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "item_id",
        "source_type",
        "title",
        "source_url",
        "license_or_terms",
        "local_path",
        "status",
        "size_bytes",
        "notes",
    ]
    with DATA_INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()

    official_statuses: dict[str, str] = {}
    for asset in OFFICIAL_ASSETS:
        source = ROOT / "assets" / asset["filename"]
        destination = OFFICIAL_ASSETS_DIR / asset["filename"]
        official_statuses[asset["filename"]] = copy_if_needed(source, destination)

    kubric_readme_path = KUBRIC_CACHE_DIR / "CoTracker3_Kubric_README.md"
    kubric_readme_status = download_if_needed(
        "https://huggingface.co/datasets/facebook/CoTracker3_Kubric/raw/main/README.md",
        kubric_readme_path,
    )

    kubric_shard_path = KUBRIC_CACHE_DIR / "0000.tar.gz"
    kubric_shard_status = download_if_needed(
        "https://huggingface.co/datasets/facebook/CoTracker3_Kubric/resolve/main/0000.tar.gz?download=true",
        kubric_shard_path,
    )
    kubric_extract_status = extract_kubric_shard_if_needed(kubric_shard_path, KUBRIC_SAMPLE_DIR)

    internet_archive_statuses: dict[str, str] = {}
    for video in INTERNET_ARCHIVE_VIDEOS:
        destination = INTERNET_ARCHIVE_DIR / video["filename"]
        url = f"https://archive.org/download/{video['identifier']}/{video['filename']}"
        internet_archive_statuses[video["filename"]] = download_if_needed(url, destination)

    rows = build_manifest_rows(
        official_statuses=official_statuses,
        kubric_readme_status=kubric_readme_status,
        kubric_shard_status=kubric_shard_status,
        kubric_extract_status=kubric_extract_status,
        internet_archive_statuses=internet_archive_statuses,
    )
    write_manifest(rows)

    print("Data collection complete.")
    print(f"Inventory written to: {relative_path(DATA_INVENTORY)}")
    print(f"Internet Archive videos tracked: {len(INTERNET_ARCHIVE_VIDEOS)}")


if __name__ == "__main__":
    main()
