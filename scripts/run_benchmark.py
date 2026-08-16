import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from p_audio.train_f1 import train_f1_one_class_svm
    from p_audio.f1_zero_day.pretrained_detector import PretrainedAntiSpoofDetector
except ImportError:
    from src.p_audio.train_f1 import train_f1_one_class_svm
    from src.p_audio.f1_zero_day.pretrained_detector import PretrainedAntiSpoofDetector

from scripts.verify_datasets import verify_manifest
from scripts.calibrate_f1 import calibrate_f1
from scripts.calibrate_f2 import calibrate_f2
from scripts.evaluate_f1 import evaluate_f1
from scripts.evaluate_f2 import evaluate_f2


def main():
    parser = argparse.ArgumentParser(description="Master End-to-End Forensic Benchmark Runner for P-Audio")
    parser.add_argument("--mode", choices=["benchmark", "smoke-test"], default="benchmark", help="Execution mode (benchmark requires real datasets)")
    parser.add_argument("--out-dir", default="output/benchmark", help="Output directory for benchmark reports")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" DEEPTRACE P-AUDIO FORENSIC BENCHMARK RUNNER")
    print("==================================================")
    print(f" EXECUTION MODE: {args.mode.upper()}")
    print(f" OUTPUT DIRECTORY: {out_dir.resolve()}")
    print("==================================================")

    # 1. Check Pretrained Neural Detector Availability
    pretrained_det = PretrainedAntiSpoofDetector()
    pretrained_det.load_model()
    pretrained_info = {
        "model_name": pretrained_det.model_name,
        "loaded_successfully": pretrained_det.is_loaded and not pretrained_det.use_offline_fallback,
        "mode": "fine_tuned_anti_spoof_neural" if (pretrained_det.is_loaded and not pretrained_det.use_offline_fallback) else "offline_acoustic_baseline"
    }

    print(f"\n[1/5] Pretrained Neural Detector Check:")
    print(f"  Model: {pretrained_info['model_name']}")
    print(f"  Neural Checkpoint Loaded: {pretrained_info['loaded_successfully']}")
    print(f"  Operating Mode: {pretrained_info['mode']}")

    # 2. Check Dataset Health & Manifests
    vctk_res = verify_manifest(Path("data/manifests/vctk_manifest.csv"))
    la_res = verify_manifest(Path("data/manifests/asvspoof_la_manifest.csv"))
    pa_res = verify_manifest(Path("data/manifests/asvspoof_pa_manifest.csv"))

    real_datasets_available = vctk_res.get("valid", False) or la_res.get("valid", False) or pa_res.get("valid", False)

    if args.mode == "benchmark" and not real_datasets_available:
        print("\n==================================================")
        print(" DEEPTRACE P-AUDIO BENCHMARK REPORT")
        print("==================================================")
        print(" STATUS: BENCHMARK NOT COMPLETED")
        print("\n REASON:")
        print(" Real public benchmark datasets (VCTK / ASVspoof 2019 LA / ASVspoof 2019 PA)")
        print(" were not found in data/datasets/ or data/manifests/.")
        print("\n DATASET PREPARATION INSTRUCTIONS:")
        print(" 1. Run automated dataset downloader:")
        print("    uv run python scripts/prepare_datasets.py --download")
        print(" 2. Or download official datasets manually:")
        print("    - VCTK: https://doi.org/10.7488/ds/2645")
        print("    - ASVspoof 2019: https://www.asvspoof.org/index2019.html")
        print(" 3. Re-run benchmark:")
        print("    uv run python scripts/run_benchmark.py --mode benchmark")
        print("==================================================")

        summary_data = {
            "status": "BENCHMARK NOT COMPLETED",
            "reason": "Real public benchmark datasets (VCTK / ASVspoof) not installed.",
            "mode": args.mode,
            "pretrained_model": pretrained_info,
            "synthetic_fallback_used": False,
            "instructions": "Run 'uv run python scripts/prepare_datasets.py' to download real datasets."
        }
        with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        report_md = f"""# DeepTrace P-Audio Forensic Benchmark Report

> [!WARNING]
> **STATUS**: BENCHMARK NOT COMPLETED
>
> Real public benchmark datasets (VCTK / ASVspoof 2019 LA / PA) are not installed in `data/datasets/`.
> The benchmark pipeline strictly refrains from fabricating test results or using synthetic placeholder audio as real benchmark data.

## Setup Instructions

To execute full benchmark evaluation:
1. Prepare datasets: `uv run python scripts/prepare_datasets.py --download`
2. Re-run benchmark: `uv run python scripts/run_benchmark.py --mode benchmark`

## Pretrained Neural Detector Status
- **Model Name**: `{pretrained_info['model_name']}`
- **Neural Loaded**: `{pretrained_info['loaded_successfully']}`
- **Operating Mode**: `{pretrained_info['mode']}`
"""
        with open(out_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        sys.exit(1)

    # 3. Training One-Class SVM on Genuine Speech
    print("\n[2/5] Training F1 One-Class SVM on Genuine Real Speech...")
    try:
        train_f1_meta = train_f1_one_class_svm(mode=args.mode)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        sys.exit(1)

    # 4. Calibrating F1 and F2 Thresholds
    print("\n[3/5] Calibrating F1 & F2 Thresholds on Validation Split...")
    cal_f1_meta = calibrate_f1(mode=args.mode)
    cal_f2_meta = calibrate_f2(mode=args.mode)

    # 5. Evaluating F1 and F2 on Test Split
    print("\n[4/5] Evaluating F1 & F2 on Test Split...")
    f1_eval = evaluate_f1(output_dir=str(out_dir), mode=args.mode)
    f2_eval = evaluate_f2(output_dir=str(out_dir), mode=args.mode)

    # 6. Build Master Summary and Markdown Report
    print("\n[5/5] Generating Master Benchmark Summary and Markdown Report...")
    summary = {
        "status": "BENCHMARK COMPLETED",
        "mode": args.mode,
        "pretrained_model": pretrained_info,
        "f1_training": train_f1_meta,
        "f1_calibration": cal_f1_meta,
        "f2_calibration": cal_f2_meta,
        "f1_test_evaluation": f1_eval,
        "f2_test_evaluation": f2_eval,
        "synthetic_fallback_used": (args.mode != "benchmark")
    }

    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = f"""# DeepTrace P-Audio Forensic Benchmark Report

**Status**: {summary['status']}  
**Mode**: `{args.mode.upper()}`  
**Pretrained Model Mode**: `{pretrained_info['mode']}`  
**Synthetic Fallback**: `{'DISABLED' if args.mode == 'benchmark' else 'ENABLED (SMOKE TEST)'}`

## F1 — Zero-Day / Synthetic Speech Detection Benchmark

- **Dataset**: `{f1_eval.get('dataset', 'ASVspoof 2019 LA')}`
- **Test Samples**: `{f1_eval.get('total_test_samples', 0)}`
- **Accuracy**: `{f1_eval.get('accuracy', 0.0)*100:.2f}%`
- **Precision**: `{f1_eval.get('precision', 0.0):.4f}`
- **Recall**: `{f1_eval.get('recall', 0.0):.4f}`
- **F1 Score**: `{f1_eval.get('f1_score', 0.0):.4f}`
- **Equal Error Rate (EER)**: `{f1_eval.get('eer', 0.0)*100:.2f}%`
- **Known Attacks (A01-A06) Accuracy**: `{f1_eval.get('known_attack_accuracy', 0.0)*100:.2f}%`
- **Zero-Day Held-Out (A07-A19) Accuracy**: `{f1_eval.get('zero_day_held_out_accuracy', 0.0)*100:.2f}%`

## F2 — Replay / Physical-World Re-recording Detection Benchmark

- **Dataset**: `{f2_eval.get('dataset', 'ASVspoof 2019 PA')}`
- **Test Samples**: `{f2_eval.get('total_test_samples', 0)}`
- **Accuracy**: `{f2_eval.get('accuracy', 0.0)*100:.2f}%`
- **Threshold Status**: `{f2_eval.get('threshold_status', 'NOT_CALIBRATED')}`

==================================================
BENCHMARK EXECUTION COMPLETE
==================================================
"""
    with open(out_dir / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("==================================================")
    print(" BENCHMARK COMPLETE")
    print(f" Reports saved to {out_dir.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
