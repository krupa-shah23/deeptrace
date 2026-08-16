import argparse
import json
import os
import sys
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from scripts.build_manifests import build_vctk_manifest, build_asvspoof_la_manifest, build_asvspoof_pa_manifest


DATASET_SOURCES = {
    "vctk": {
        "name": "CSTR VCTK Corpus",
        "target_dir": "data/datasets/vctk",
        "manifest": "data/manifests/vctk_manifest.csv",
        "info_url": "https://doi.org/10.7488/ds/2645",
        "download_url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    },
    "asvspoof2019_la": {
        "name": "ASVspoof 2019 Logical Access (LA)",
        "target_dir": "data/datasets/asvspoof2019_la",
        "manifest": "data/manifests/asvspoof_la_manifest.csv",
        "info_url": "https://www.asvspoof.org/index2019.html",
        "download_url": "https://zenodo.org/record/4835108/files/LA.zip"
    },
    "asvspoof2019_pa": {
        "name": "ASVspoof 2019 Physical Access (PA)",
        "target_dir": "data/datasets/asvspoof2019_pa",
        "manifest": "data/manifests/asvspoof_pa_manifest.csv",
        "info_url": "https://www.asvspoof.org/index2019.html",
        "download_url": "https://zenodo.org/record/4835108/files/PA.zip"
    }
}


def prepare_dataset(dataset_key: str, download: bool = False, link_path: str = None) -> bool:
    """Prepares and verifies specified public dataset."""
    info = DATASET_SOURCES.get(dataset_key)
    if not info:
        print(f"[Prepare Dataset] Unknown dataset key '{dataset_key}'. Choose from: {list(DATASET_SOURCES.keys())}")
        return False

    target_dir = Path(info["target_dir"])
    manifest_csv = Path(info["manifest"])

    print(f"\n--- Preparing {info['name']} ---")
    print(f"Target Directory: {target_dir.resolve()}")

    if link_path:
        src = Path(link_path)
        if not src.exists():
            print(f"[ERROR] Specified link path does not exist: {src}")
            return False
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_dir.exists():
            try:
                os.symlink(src.resolve(), target_dir.resolve(), target_is_directory=True)
                print(f"[Prepare Dataset] Linked {src} -> {target_dir}")
            except Exception as e:
                print(f"[Prepare Dataset] Could not create symlink ({e}). Copying reference structure instead.")

    if not target_dir.exists() or len(list(target_dir.rglob("*"))) == 0:
        if download:
            print(f"[Prepare Dataset] Downloading {info['name']} from {info['download_url']}...")
            target_dir.mkdir(parents=True, exist_ok=True)
            zip_path = target_dir.parent / f"{dataset_key}.zip"
            try:
                urllib.request.urlretrieve(info['download_url'], zip_path)
                print(f"[Prepare Dataset] Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print(f"[Prepare Dataset] Download and extraction complete.")
            except Exception as e:
                print(f"[ERROR] Failed to download dataset automatically: {e}")
                print(f"Please manually download {info['name']} from {info['info_url']} and extract to {target_dir.resolve()}")
                return False
        else:
            print(f"[STATUS] {info['name']} is NOT currently installed in '{target_dir}'.")
            print(f"  To install: Download from {info['info_url']} and extract into {target_dir.resolve()}")
            print(f"  Or run: uv run python scripts/prepare_datasets.py --dataset {dataset_key} --download")
            return False

    # Build manifest
    count = 0
    if dataset_key == "vctk":
        count = build_vctk_manifest(target_dir, manifest_csv)
    elif dataset_key == "asvspoof2019_la":
        count = build_asvspoof_la_manifest(target_dir, manifest_csv)
    elif dataset_key == "asvspoof2019_pa":
        count = build_asvspoof_pa_manifest(target_dir, manifest_csv)

    return count > 0


def main():
    parser = argparse.ArgumentParser(description="DeepTrace P-Audio Public Dataset Preparation & Setup Script")
    parser.add_argument("--dataset", choices=["vctk", "asvspoof2019_la", "asvspoof2019_pa", "all"], default="all", help="Dataset to prepare")
    parser.add_argument("--download", action="store_true", help="Automatically attempt downloading official release zips")
    parser.add_argument("--link", help="Path to local folder containing downloaded dataset")

    args = parser.parse_args()

    print("==================================================")
    print(" DEEPTRACE P-AUDIO DATASET PREPARATION & SETUP")
    print("==================================================")

    targets = list(DATASET_SOURCES.keys()) if args.dataset == "all" else [args.dataset]
    results = {}
    for t in targets:
        results[t] = prepare_dataset(t, download=args.download, link_path=args.link)

    print("\n==================================================")
    print(" DATASET STATUS SUMMARY")
    print("==================================================")
    all_ready = True
    for k, v in results.items():
        name = DATASET_SOURCES[k]["name"]
        status = "READY" if v else "NOT INSTALLED"
        if not v:
            all_ready = False
        print(f" - {name:<38}: [{status}]")

    if not all_ready:
        print("\nNote: Public benchmark datasets (VCTK / ASVspoof) are required for full benchmark mode.")
        print("Run with '--download' or follow instructions above to download official datasets.")
        sys.exit(1)
    else:
        print("\nAll target datasets are prepared and manifests are updated!")
        sys.exit(0)


if __name__ == "__main__":
    main()
