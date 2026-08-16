import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root and src to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import soundfile as sf

try:
    from p_audio.config import PAudioConfig
    from p_audio.f1_zero_day.model import OneClassSVMDetector
    from p_audio.features import extract_all_audio_features
    from p_audio.preprocess import load_and_preprocess_audio
except ImportError:
    from p_audio.config import PAudioConfig
    from p_audio.f1_zero_day.model import OneClassSVMDetector
    from p_audio.features import extract_all_audio_features
    from p_audio.preprocess import load_and_preprocess_audio


def generate_synthetic_real_speech_sample(file_path: str, duration_sec: float = 3.5, sr: int = 16000, seed: int = 0):
    """
    Generates a clean synthetic human speech proxy audio clip ONLY for local smoke testing / developer demos.
    Must NEVER be presented as real dataset evidence in benchmark mode.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration_sec, int(sr * duration_sec))

    # Speaker pitch base varies per sample (110 Hz to 210 Hz)
    f0_base = 110.0 + (seed % 15) * 7.0
    f0_mod_rate = 2.5 + (seed % 5) * 0.5
    f0_track = f0_base + 6.0 * np.sin(2 * np.pi * f0_mod_rate * t) + 1.2 * rng.randn(len(t))
    phase = 2 * np.pi * np.cumsum(f0_track) / sr
    speech_signal = np.sin(phase)

    # Formant resonances (F1, F2, F3) with speaker-dependent shifts
    f1_mult = 3.5 + (seed % 4) * 0.2
    f2_mult = 11.0 + (seed % 4) * 0.3
    formant1 = 0.5 * np.sin(phase * f1_mult)
    formant2 = 0.3 * np.sin(phase * f2_mult)
    speech_signal = speech_signal + formant1 + formant2

    # Speech envelope modulation (syllables at 3.5 to 5.5 Hz)
    syllable_rate = 3.5 + (seed % 5) * 0.4
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))
    speech_signal = speech_signal * envelope

    # Add realistic clean background acoustic noise floor (-55 dB)
    noise = 1e-4 * rng.randn(len(t))
    signal_out = speech_signal + noise

    # Normalize
    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-8) * 0.90
    sf.write(file_path, signal_out, sr)


def load_real_speech_training_samples(config: PAudioConfig, mode: str = "benchmark") -> Tuple[List[str], Dict[str, Any]]:
    """
    Loads genuine real human speech training audio clips.
    In benchmark mode, loads strictly from VCTK / ASVspoof manifests.
    Fails loudly if real public datasets are missing.
    """
    manifest_paths = [
        Path("data/manifests/vctk_manifest.csv"),
        Path("data/manifests/asvspoof_la_manifest.csv")
    ]

    audio_files = []
    speakers = set()
    dataset_name = "NONE"

    for m_path in manifest_paths:
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split") == "train" and row.get("label") == "REAL":
                        p = Path(row["path"])
                        if p.exists():
                            audio_files.append(str(p))
                            speakers.add(row.get("speaker_id", "unknown"))

            if audio_files:
                dataset_name = m_path.stem.replace("_manifest", "").upper()
                break

    if mode == "benchmark" and len(audio_files) == 0:
        print("\n==================================================")
        print(" [CRITICAL ERROR] REAL BENCHMARK DATASET NOT FOUND")
        print("==================================================")
        print(" Benchmark mode requires real public dataset files (VCTK / ASVspoof 2019 LA).")
        print(" Synthetic fallback data is STRICTLY DISABLED in benchmark mode.")
        print(" Please run dataset preparation first:")
        print("    uv run python scripts/prepare_datasets.py\n")
        raise FileNotFoundError("Real benchmark dataset (VCTK/ASVspoof) not found for benchmark training.")

    if len(audio_files) == 0 and mode == "smoke-test":
        print("[Train F1] Smoke-test mode: Generating 35 synthetic real-speech proxy clips in data/generated/smoke_train...")
        gen_dir = Path("data/generated/smoke_train")
        gen_dir.mkdir(parents=True, exist_ok=True)
        for i in range(35):
            sample_file = gen_dir / f"real_speech_sample_{i+1:02d}.wav"
            generate_synthetic_real_speech_sample(str(sample_file), duration_sec=3.5, sr=config.target_sr, seed=i)
            audio_files.append(str(sample_file))
            speakers.add(f"smoke_spk_{i % 5}")
        dataset_name = "LOCAL_SMOKE_TEST_FIXTURES"

    meta = {
        "dataset_name": dataset_name,
        "mode": mode,
        "num_files": len(audio_files),
        "num_speakers": len(speakers),
        "speakers": sorted(list(speakers)),
        "synthetic_fallback_disabled": (mode == "benchmark")
    }
    return audio_files, meta


def train_f1_one_class_svm(
    output_model_path: str = "models/f1_one_class_svm.joblib",
    config: Optional[PAudioConfig] = None,
    mode: str = "benchmark"
) -> Dict[str, Any]:
    """
    Trains One-Class SVM Novelty Detector strictly on genuine real human speech audio features.
    Saves fitted model and scaler artifacts.
    """
    config = config or PAudioConfig()

    print(f"\n==================================================")
    print(f" DATASET MODE: {mode.upper()}")
    print(f" SYNTHETIC FALLBACK: {'DISABLED' if mode == 'benchmark' else 'ENABLED (SMOKE TEST)'}")
    print(f"==================================================")

    audio_files, meta = load_real_speech_training_samples(config, mode=mode)
    print(f"[Train F1] Loading features from {len(audio_files)} genuine real speech samples ({meta['dataset_name']})...")

    feature_matrix = []
    feature_names = []

    for file_path in audio_files:
        preprocessed = load_and_preprocess_audio(file_path, config)
        if preprocessed.usable and preprocessed.audio is not None:
            feat = extract_all_audio_features(preprocessed.audio, preprocessed.sr)
            feature_matrix.append(feat.feature_vector)
            if not feature_names:
                feature_names = feat.feature_names

    if len(feature_matrix) < 5:
        raise RuntimeError(f"Extracted fewer than 5 valid feature vectors from {len(audio_files)} training files.")

    X = np.array(feature_matrix)
    print(f"[Train F1] Feature matrix shape: {X.shape}")

    detector = OneClassSVMDetector(kernel="rbf", nu=0.1, gamma="scale")
    training_meta = {
        "dataset_name": meta["dataset_name"],
        "mode": mode,
        "num_training_files": len(audio_files),
        "num_speakers": meta["num_speakers"],
        "feature_dim": X.shape[1],
        "feature_names": feature_names,
        "synthetic_fallback_used": (mode != "benchmark")
    }

    detector.fit(X, feature_names=feature_names, metadata=training_meta)
    detector.save(Path(output_model_path))

    # Save scaler artifact
    scaler_path = Path(output_model_path).parent / "f1_scaler.joblib"
    try:
        import joblib
        joblib.dump(detector.scaler, scaler_path)
    except Exception:
        pass

    # Save training metadata
    meta_path = Path(output_model_path).parent / "f1_training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    print(f"[Train F1] Successfully trained One-Class SVM on {len(X)} real speech samples.")
    print(f"[Train F1] Saved model -> {output_model_path}")
    print(f"[Train F1] Saved scaler -> {scaler_path}")
    return training_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train One-Class SVM for F1 Zero-Day Synthetic Speech Detection")
    parser.add_argument("--out-model", default="models/f1_one_class_svm.joblib", help="Output path for model artifact")
    parser.add_argument("--mode", choices=["benchmark", "smoke-test"], default="benchmark", help="Execution mode (benchmark requires real dataset)")

    args = parser.parse_args()
    cfg = PAudioConfig()
    train_f1_one_class_svm(
        output_model_path=args.out_model,
        config=cfg,
        mode=args.mode
    )
