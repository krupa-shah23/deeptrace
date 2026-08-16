import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def build_vctk_manifest(vctk_dir: Path, output_csv: Path) -> int:
    """
    Parses VCTK directory structure and speaker-info.txt to generate vctk_manifest.csv.
    """
    if not vctk_dir.exists():
        print(f"[Build Manifest] VCTK directory not found at {vctk_dir}")
        return 0

    wav_files = list(vctk_dir.rglob("*.flac")) + list(vctk_dir.rglob("*.wav"))
    if not wav_files:
        print(f"[Build Manifest] No WAV/FLAC audio files found in {vctk_dir}")
        return 0

    # Parse speaker-info.txt if available
    speaker_info_file = vctk_dir / "speaker-info.txt"
    speaker_meta = {}
    if speaker_info_file.exists():
        with open(speaker_info_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    spk_id = f"p{parts[0]}"
                    speaker_meta[spk_id] = {
                        "age": parts[1] if len(parts) > 1 else "NA",
                        "gender": parts[2] if len(parts) > 2 else "NA",
                        "accents": parts[3] if len(parts) > 3 else "NA"
                    }

    # Extract unique speakers for speaker-disjoint split
    speakers = sorted(list(set(f.name.split("_")[0] for f in wav_files if "_" in f.name)))
    if not speakers:
        speakers = sorted(list(set(f.parent.name for f in wav_files)))

    num_speakers = len(speakers)
    train_end = int(0.70 * num_speakers)
    val_end = int(0.85 * num_speakers)

    train_spks = set(speakers[:train_end])
    val_spks = set(speakers[train_end:val_end])
    test_spks = set(speakers[val_end:])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "path", "dataset", "speaker_id", "label",
            "split", "duration_sec", "sample_rate", "language", "source_type"
        ])
        for file_path in wav_files:
            spk_id = file_path.name.split("_")[0] if "_" in file_path.name else file_path.parent.name
            if spk_id in train_spks:
                split = "train"
            elif spk_id in val_spks:
                split = "val"
            else:
                split = "test"

            writer.writerow([
                file_path.stem,
                str(file_path.resolve()),
                "VCTK",
                spk_id,
                "REAL",
                split,
                "NA",
                16000,
                "English",
                "BONAFIDE_SPEECH"
            ])
            count += 1

    print(f"[Build Manifest] Built VCTK manifest ({count} samples, {num_speakers} speakers) -> {output_csv}")
    return count


def build_asvspoof_la_manifest(la_dir: Path, output_csv: Path) -> int:
    """
    Parses official ASVspoof 2019 LA protocol files (train, dev, eval) to generate asvspoof_la_manifest.csv.
    """
    if not la_dir.exists():
        print(f"[Build Manifest] ASVspoof 2019 LA directory not found at {la_dir}")
        return 0

    protocol_files = list(la_dir.rglob("ASVspoof2019.LA.cm.*.txt")) + list(la_dir.rglob("*.txt"))
    if not protocol_files:
        print(f"[Build Manifest] No ASVspoof 2019 LA protocol files found in {la_dir}")
        return 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "sample_id", "path", "dataset", "speaker_id", "label",
            "attack_id", "split", "is_zero_day", "duration_sec", "sample_rate"
        ])

        for p_file in protocol_files:
            # Determine split from filename
            p_name = p_file.name.lower()
            if "train" in p_name:
                split = "train"
            elif "dev" in p_name:
                split = "val"
            elif "eval" in p_name:
                split = "test"
            else:
                continue

            with open(p_file, "r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # Official format: SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY/LABEL
                    # e.g., LA_0079 LA_E_2834763 - A09 spoof
                    # or LA_0079 LA_E_2834763 - - bonafide
                    spk_id = parts[0]
                    audio_stem = parts[1]
                    attack_id = parts[3] if len(parts) >= 4 else "-"
                    label_raw = parts[-1].lower()

                    label = "REAL" if label_raw in ("bonafide", "real") else "SYNTHETIC"
                    is_zero_day = (split == "test" and attack_id not in ("-", "A01", "A02", "A03", "A04", "A05", "A06"))

                    # Locate actual audio file
                    audio_matches = list(la_dir.rglob(f"{audio_stem}.*"))
                    file_path_str = str(audio_matches[0].resolve()) if audio_matches else str(la_dir / f"{audio_stem}.flac")

                    writer.writerow([
                        audio_stem,
                        file_path_str,
                        "ASVspoof2019LA",
                        spk_id,
                        label,
                        attack_id,
                        split,
                        is_zero_day,
                        "NA",
                        16000
                    ])
                    count += 1

    print(f"[Build Manifest] Built ASVspoof 2019 LA manifest ({count} samples) -> {output_csv}")
    return count


def build_asvspoof_pa_manifest(pa_dir: Path, output_csv: Path) -> int:
    """
    Parses official ASVspoof 2019 PA protocol files (train, dev, eval) to generate asvspoof_pa_manifest.csv.
    """
    if not pa_dir.exists():
        print(f"[Build Manifest] ASVspoof 2019 PA directory not found at {pa_dir}")
        return 0

    protocol_files = list(pa_dir.rglob("ASVspoof2019.PA.cm.*.txt")) + list(pa_dir.rglob("*.txt"))
    if not protocol_files:
        print(f"[Build Manifest] No ASVspoof 2019 PA protocol files found in {pa_dir}")
        return 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "sample_id", "path", "dataset", "speaker_id", "label",
            "replay_config_id", "split", "duration_sec", "sample_rate"
        ])

        for p_file in protocol_files:
            p_name = p_file.name.lower()
            if "train" in p_name:
                split = "train"
            elif "dev" in p_name:
                split = "val"
            elif "eval" in p_name:
                split = "test"
            else:
                continue

            with open(p_file, "r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    spk_id = parts[0]
                    audio_stem = parts[1]
                    replay_config = parts[3] if len(parts) >= 4 else "-"
                    label_raw = parts[-1].lower()

                    label = "REAL" if label_raw in ("bonafide", "real") else "REPLAY"

                    audio_matches = list(pa_dir.rglob(f"{audio_stem}.*"))
                    file_path_str = str(audio_matches[0].resolve()) if audio_matches else str(pa_dir / f"{audio_stem}.flac")

                    writer.writerow([
                        audio_stem,
                        file_path_str,
                        "ASVspoof2019PA",
                        spk_id,
                        label,
                        replay_config,
                        split,
                        "NA",
                        16000
                    ])
                    count += 1

    print(f"[Build Manifest] Built ASVspoof 2019 PA manifest ({count} samples) -> {output_csv}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Build Machine-Readable CSV Manifests for Forensic Audio Datasets")
    parser.add_argument("--vctk-dir", default="data/datasets/vctk", help="Path to VCTK dataset root")
    parser.add_argument("--asvspoof-la-dir", default="data/datasets/asvspoof2019_la", help="Path to ASVspoof 2019 LA dataset root")
    parser.add_argument("--asvspoof-pa-dir", default="data/datasets/asvspoof2019_pa", help="Path to ASVspoof 2019 PA dataset root")
    parser.add_argument("--out-dir", default="data/manifests", help="Output directory for manifests")

    args = parser.parse_args()
    out_path = Path(args.out_dir)

    print("==================================================")
    print(" DEEPTRACE P-AUDIO DATASET MANIFEST BUILDER")
    print("==================================================")
    build_vctk_manifest(Path(args.vctk_dir), out_path / "vctk_manifest.csv")
    build_asvspoof_la_manifest(Path(args.asvspoof_la_dir), out_path / "asvspoof_la_manifest.csv")
    build_asvspoof_pa_manifest(Path(args.asvspoof_pa_dir), out_path / "asvspoof_pa_manifest.csv")
    print("Manifest generation complete.")


if __name__ == "__main__":
    main()
