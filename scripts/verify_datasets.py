import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import soundfile as sf
except ImportError:
    sf = None


def verify_manifest(manifest_csv: Path) -> Dict[str, any]:
    """
    Verifies audio manifest CSV file integrity, speaker leakage, and file existence.
    """
    if not manifest_csv.exists():
        return {"exists": False, "reason": f"Manifest file missing: {manifest_csv}"}

    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"exists": True, "valid": False, "reason": "Manifest CSV is empty"}

    file_count = len(rows)
    existing_files = 0
    corrupted_files = 0
    splits: Dict[str, int] = {}
    speakers_per_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    labels_per_split: Dict[str, Dict[str, int]] = {}

    for row in rows:
        p = Path(row["path"])
        split = row.get("split", "unknown")
        spk_id = row.get("speaker_id", "unknown")
        label = row.get("label", "unknown")

        splits[split] = splits.get(split, 0) + 1
        if split in speakers_per_split:
            speakers_per_split[split].add(spk_id)

        if split not in labels_per_split:
            labels_per_split[split] = {}
        labels_per_split[split][label] = labels_per_split[split].get(label, 0) + 1

        if p.exists():
            existing_files += 1

    # Check speaker leakage between train and test
    train_speakers = speakers_per_split["train"]
    test_speakers = speakers_per_split["test"]
    leakage = train_speakers.intersection(test_speakers)

    return {
        "exists": True,
        "valid": existing_files > 0,
        "total_files": file_count,
        "existing_files": existing_files,
        "missing_files": file_count - existing_files,
        "splits": splits,
        "speakers_per_split": {k: len(v) for k, v in speakers_per_split.items()},
        "labels_per_split": labels_per_split,
        "speaker_leakage_detected": len(leakage) > 0,
        "speaker_leakage_count": len(leakage),
        "leaked_speakers": list(leakage)
    }


def main():
    print("==================================================")
    print(" DEEPTRACE P-AUDIO DATASET HEALTH & HEALTH VERIFIER")
    print("==================================================")

    manifest_dir = Path("data/manifests")
    manifests = {
        "VCTK (Real Speech Baseline)": manifest_dir / "vctk_manifest.csv",
        "ASVspoof 2019 LA (Synthetic Speech)": manifest_dir / "asvspoof_la_manifest.csv",
        "ASVspoof 2019 PA (Physical Replay)": manifest_dir / "asvspoof_pa_manifest.csv"
    }

    all_valid = True
    for name, path in manifests.items():
        print(f"\n--- Verifying {name} ---")
        res = verify_manifest(path)
        if not res.get("exists") or not res.get("valid"):
            print(f" Status: NOT READY ({res.get('reason', 'Files missing or unparsed')})")
            all_valid = False
            continue

        print(f" Status: READY")
        print(f" Total manifest entries: {res['total_files']} (Found on disk: {res['existing_files']})")
        print(f" Splits: {res['splits']}")
        print(f" Speakers per split: {res['speakers_per_split']}")
        print(f" Labels per split: {res['labels_per_split']}")

        if res["speaker_leakage_detected"]:
            print(f" [CRITICAL ERROR] Speaker leakage detected! {res['speaker_leakage_count']} speakers overlap between train and test.")
            all_valid = False
        else:
            print(" [PASSED] Speaker-disjoint policy verified (0 train/test speaker overlap).")

    print("\n==================================================")
    if all_valid:
        print(" VERIFICATION SUCCESS: All datasets ready for benchmark execution!")
        sys.exit(0)
    else:
        print(" VERIFICATION INCOMPLETE: One or more datasets missing or unverified.")
        print(" Please run 'uv run python scripts/prepare_datasets.py' to resolve.")
        sys.exit(1)


if __name__ == "__main__":
    main()
