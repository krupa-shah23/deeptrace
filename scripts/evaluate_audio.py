import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root and src to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from p_audio.config import PAudioConfig
    from p_audio.pipeline import PAudioPipeline
except ImportError:
    from p_audio.config import PAudioConfig
    from p_audio.pipeline import PAudioPipeline


def discover_audio_files(directory: str) -> List[Path]:
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    exts = {"*.wav", "*.mp3", "*.flac", "*.m4a", "*.mp4", "*.mkv"}
    files = []
    for ext in exts:
        files.extend(list(dir_path.glob(ext)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="DeepTrace P-Audio Forensic Evaluation Script")
    parser.add_argument("--real-dir", default="data/audio/real", help="Directory containing genuine real speech files")
    parser.add_argument("--synthetic-dir", default="data/audio/synthetic", help="Directory containing synthetic / deepfake audio files")
    parser.add_argument("--replay-dir", default="data/audio/replay", help="Directory containing physical re-recorded audio files")
    parser.add_argument("--out-json", default="output/audio_evaluation_report.json", help="Output path for evaluation report JSON")

    args = parser.parse_args()

    config = PAudioConfig()
    pipeline = PAudioPipeline(config)

    dataset_map = [
        (args.real_dir, "REAL"),
        (args.synthetic_dir, "SYNTHETIC"),
        (args.replay_dir, "REPLAY")
    ]

    evaluated_samples = []

    print("\n" + "=" * 90)
    print(" DEEPTRACE P-AUDIO FORENSIC EVALUATION REPORT")
    print("=" * 90)

    for dir_path, ground_truth in dataset_map:
        files = discover_audio_files(dir_path)
        if not files:
            print(f"[Evaluation] Category '{ground_truth}': No audio files found in '{dir_path}'")
            continue

        print(f"[Evaluation] Category '{ground_truth}': Found {len(files)} audio clips in '{dir_path}'")
        for file_path in files:
            t0 = time.time()
            res = pipeline.analyze_clip(str(file_path))
            proc_time = round(time.time() - t0, 3)

            item = {
                "filename": file_path.name,
                "ground_truth": ground_truth,
                "f1_verdict": res.get("f1_zero_day", {}).get("verdict", "INCONCLUSIVE"),
                "f1_score": res.get("f1_zero_day", {}).get("confidence", 0.0),
                "f2_pathway": res.get("f2_replay", {}).get("pathway", "INCONCLUSIVE"),
                "f2_score": res.get("f2_replay", {}).get("confidence", 0.0),
                "combined_verdict": res.get("verdict", "INCONCLUSIVE"),
                "processing_time_sec": proc_time
            }
            evaluated_samples.append(item)

    print("\n" + "-" * 110)
    print(f"{'Filename':<30} | {'Ground Truth':<12} | {'F1 Verdict':<18} | {'F1 Score':<8} | {'F2 Pathway':<20} | {'F2 Score':<8} | {'Time (s)'}")
    print("-" * 110)

    for sample in evaluated_samples:
        print(f"{sample['filename']:<30} | {sample['ground_truth']:<12} | {sample['f1_verdict']:<18} | {sample['f1_score']:<8.4f} | {sample['f2_pathway']:<20} | {sample['f2_score']:<8.4f} | {sample['processing_time_sec']:<7.3f}")

    print("-" * 110)

    # Compute confusion matrix & metrics if sufficient samples exist
    if len(evaluated_samples) >= 5:
        tp, fp, fn, tn = 0, 0, 0, 0
        for sample in evaluated_samples:
            gt = sample["ground_truth"]
            pred = sample["combined_verdict"]

            if gt in ("SYNTHETIC", "REPLAY"):
                if pred in ("KNOWN_SYNTHETIC", "UNKNOWN_SYNTHETIC", "REPLAYED_RECORDING"):
                    tp += 1
                else:
                    fn += 1
            elif gt == "REAL":
                if pred == "REAL":
                    tn += 1
                else:
                    fp += 1

        total = tp + fp + fn + tn
        acc = (tp + tn) / float(total) if total > 0 else 0.0
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0

        metrics = {
            "total_samples": len(evaluated_samples),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
        }
        print(f"\n[Metrics] Evaluated {len(evaluated_samples)} samples:")
        print(f"  Accuracy:  {acc:.2%}")
        print(f"  Precision: {prec:.2%}")
        print(f"  Recall:    {rec:.2%}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    else:
        metrics = {
            "total_samples": len(evaluated_samples),
            "status": "INSUFFICIENT SAMPLES TO CALCULATE STATISTICAL METRICS — DATASET REQUIRED"
        }
        print("\n[Metrics] Note: Insufficient labeled samples found to calculate statistical metrics (min 5 required).")
        print("          IMPLEMENTED — VALIDATION DATA REQUIRED.")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "samples": evaluated_samples
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {out_path}")


if __name__ == "__main__":
    main()
