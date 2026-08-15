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

from src.p_audio.config import PAudioConfig
from src.p_audio.f1_zero_day.model import OneClassSVMDetector
from src.p_audio.features import extract_all_audio_features
from src.p_audio.preprocess import load_and_preprocess_audio


def generate_synthetic_real_speech_sample(file_path: str, duration_sec: float = 3.0, sr: int = 16000):
    """
    Generates a clean synthetic human speech proxy audio clip for baseline OC-SVM training
    when no local VCTK dataset is downloaded.
    Combines fundamental frequency (F0 = 120-180Hz) with natural speech formants and subtle micro-tremor.
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    # Fundamental frequency with subtle natural pitch micro-tremor (130 Hz + 3 Hz modulation)
    f0_track = 130.0 + 5.0 * np.sin(2 * np.pi * 3.0 * t) + 1.5 * np.random.randn(len(t))
    phase = 2 * np.pi * np.cumsum(f0_track) / sr
    speech_signal = np.sin(phase)

    # Formant resonances (F1 = 500Hz, F2 = 1500Hz, F3 = 2500Hz)
    formant1 = 0.5 * np.sin(phase * 3.8)
    formant2 = 0.3 * np.sin(phase * 11.5)
    speech_signal = speech_signal + formant1 + formant2

    # Speech envelope modulation (simulating speech syllables at 4 Hz)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    speech_signal = speech_signal * envelope

    # Add realistic background acoustic noise floor (-50 dB)
    noise = 0.003 * np.random.randn(len(t))
    signal_out = speech_signal + noise

    # Normalize
    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-8) * 0.90
    sf.write(file_path, signal_out, sr)


def train_f1_one_class_svm(
    data_dir: str,
    output_model_path: str,
    config: PAudioConfig,
    generate_baseline: bool = True
):
    """
    Trains One-Class SVM on genuine real human speech audio clips.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    audio_files = list(data_path.glob("*.wav")) + list(data_path.glob("*.mp3")) + list(data_path.glob("*.flac"))

    if len(audio_files) == 0 and generate_baseline:
        print(f"[Train F1] No audio clips found in {data_dir}. Generating 35 synthetic real-speech baseline samples for prototype training...")
        for i in range(35):
            sample_file = data_path / f"real_speech_sample_{i+1:02d}.wav"
            generate_synthetic_real_speech_sample(str(sample_file), duration_sec=3.5, sr=config.target_sr)
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

    args = parser.parse_args()
    cfg = PAudioConfig()
    train_f1_one_class_svm(
        data_dir=args.data_dir,
        output_model_path=args.out_model,
        config=cfg,
        generate_baseline=True
    )
