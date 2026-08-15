import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import soundfile as sf

try:
    from p_audio.config import PAudioConfig
    from p_audio.f1_zero_day.model import OneClassSVMDetector
    from p_audio.features import extract_all_audio_features
    from p_audio.preprocess import load_and_preprocess_audio
except ImportError:
    from src.p_audio.config import PAudioConfig
    from src.p_audio.f1_zero_day.model import OneClassSVMDetector
    from src.p_audio.features import extract_all_audio_features
    from src.p_audio.preprocess import load_and_preprocess_audio


def generate_synthetic_real_speech_sample(file_path: str, duration_sec: float = 3.5, sr: int = 16000, seed: int = 0):
    """
    Generates a clean synthetic human speech proxy audio clip for baseline OC-SVM training
    when no local VCTK dataset is downloaded.
    Combines varied fundamental frequency (F0 = 110-210Hz), speaker formants, micro-tremor, and envelope modulation.
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


def train_f1_one_class_svm(
    data_dir: str,
    output_model_path: str,
    config: PAudioConfig,
    generate_baseline: bool = True,
    force_baseline: bool = False
):
    """
    Trains One-Class SVM on genuine real human speech audio clips.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if force_baseline:
        for old_file in data_path.glob("real_speech_sample_*.wav"):
            try:
                old_file.unlink()
            except Exception:
                pass

    audio_files = list(data_path.glob("*.wav")) + list(data_path.glob("*.mp3")) + list(data_path.glob("*.flac"))

    if (len(audio_files) == 0 or force_baseline) and generate_baseline:
        print(f"[Train F1] Generating 35 clean real-speech baseline samples in {data_dir} for prototype training...")
        for i in range(35):
            sample_file = data_path / f"real_speech_sample_{i+1:02d}.wav"
            generate_synthetic_real_speech_sample(str(sample_file), duration_sec=3.5, sr=config.target_sr, seed=i)
        audio_files = list(data_path.glob("*.wav"))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_dir} to train One-Class SVM.")

    print(f"[Train F1] Processing {len(audio_files)} genuine real speech samples...")

    feature_matrix = []
    feature_names = []

    for file_path in audio_files:
        preprocessed = load_and_preprocess_audio(str(file_path), config)
        if preprocessed.usable and preprocessed.audio is not None:
            feat = extract_all_audio_features(preprocessed.audio, preprocessed.sr)
            feature_matrix.append(feat.feature_vector)
            if not feature_names:
                feature_names = feat.feature_names

    if len(feature_matrix) < 5:
        raise RuntimeError(f"Fewer than 5 usable audio samples extracted from {data_dir}. Need more valid speech files.")

    X = np.array(feature_matrix)
    print(f"[Train F1] Feature matrix shape: {X.shape}")

    detector = OneClassSVMDetector(kernel="rbf", nu=0.1, gamma="scale")
    metadata = {
        "data_dir": str(data_dir),
        "num_files": len(audio_files),
        "sampling_rate": config.target_sr,
        "min_duration_sec": config.min_duration_sec
    }
    detector.fit(X, feature_names=feature_names, metadata=metadata)
    detector.save(Path(output_model_path))
    print(f"[Train F1] Successfully trained One-Class SVM on {len(X)} real speech samples and saved to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train One-Class SVM for F1 Zero-Day Synthetic Speech Detection")
    parser.add_argument("--data-dir", default="data/audio/real", help="Directory containing genuine real speech WAV files")
    parser.add_argument("--out-model", default="models/f1_one_class_svm.joblib", help="Output path for model artifact")

    parser.add_argument("--force", action="store_true", help="Force regeneration of baseline synthetic real-speech samples")

    args = parser.parse_args()
    cfg = PAudioConfig()
    train_f1_one_class_svm(
        data_dir=args.data_dir,
        output_model_path=args.out_model,
        config=cfg,
        generate_baseline=True,
        force_baseline=args.force
    )
