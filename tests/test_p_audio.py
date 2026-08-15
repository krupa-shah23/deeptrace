import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root and src to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pytest
import soundfile as sf

try:
    from p_audio.config import PAudioConfig
    from p_audio.f1_zero_day.detector import F1ZeroDayDetector
    from p_audio.f1_zero_day.model import OneClassSVMDetector
    from p_audio.f1_zero_day.pretrained_detector import PretrainedAntiSpoofDetector
    from p_audio.f2_replay.detector import F2ReplayDetector
    from p_audio.features import extract_all_audio_features, safe_nan_clean
    from p_audio.pipeline import PAudioPipeline
    from p_audio.preprocess import load_and_preprocess_audio
except ImportError:
    from src.p_audio.config import PAudioConfig
    from src.p_audio.f1_zero_day.detector import F1ZeroDayDetector
    from src.p_audio.f1_zero_day.model import OneClassSVMDetector
    from src.p_audio.f1_zero_day.pretrained_detector import PretrainedAntiSpoofDetector
    from src.p_audio.f2_replay.detector import F2ReplayDetector
    from src.p_audio.features import extract_all_audio_features, safe_nan_clean
    from src.p_audio.pipeline import PAudioPipeline
    from src.p_audio.preprocess import load_and_preprocess_audio


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def real_speech_wav(temp_dir):
    """Generates a 3.5-second speech-like audio clip (real speech proxy with formants, harmonics, micro-tremor)."""
    file_path = temp_dir / "real_speech.wav"
    sr = 16000
    t = np.linspace(0, 3.5, int(sr * 3.5))
    f0 = 135.0 + 8.0 * np.sin(2 * np.pi * 3.0 * t) + 1.5 * np.random.randn(len(t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    speech = np.sin(phase) + 0.4 * np.sin(3.5 * phase)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    y = speech * envelope + 0.002 * np.random.randn(len(t))
    y = y / np.max(np.abs(y)) * 0.90
    sf.write(file_path, y, sr)
    return str(file_path)


@pytest.fixture
def synthetic_speech_wav(temp_dir):
    """Generates a 3.5-second synthetic speech proxy (flat pitch, zero micro-tremor, vocoder high-frequency artifact)."""
    file_path = temp_dir / "synthetic_speech.wav"
    sr = 16000
    t = np.linspace(0, 3.5, int(sr * 3.5))
    # Unnatural constant pitch without micro-tremor
    f0 = 150.0 * np.ones_like(t)
    phase = 2 * np.pi * f0 * t
    speech = np.sin(phase)
    # High frequency vocoder Buzz artifact (> 6 kHz)
    buzz = 0.25 * np.sin(2 * np.pi * 7500 * t)
    y = speech + buzz
    y = y / np.max(np.abs(y)) * 0.90
    sf.write(file_path, y, sr)
    return str(file_path)


@pytest.fixture
def replayed_wav(temp_dir):
    """Generates a 3.5-second re-recorded audio clip with simulated room reverberation (T60) and elevated noise floor."""
    file_path = temp_dir / "replayed.wav"
    sr = 16000
    t = np.linspace(0, 3.5, int(sr * 3.5))
    dry_speech = np.sin(2 * np.pi * 150 * t) * 0.5
    # Room reverberation tail simulation
    reverb_impulse = np.exp(-4.0 * np.linspace(0, 0.6, int(sr * 0.6)))
    reverberant = np.convolve(dry_speech, reverb_impulse, mode="same")
    noise = 0.04 * np.random.randn(len(reverberant))  # elevated noise floor (-28 dB)
    replayed_signal = reverberant + noise
    replayed_signal = replayed_signal / np.max(np.abs(replayed_signal)) * 0.90
    sf.write(file_path, replayed_signal, sr)
    return str(file_path)


@pytest.fixture
def short_speech_wav(temp_dir):
    """Generates a 1.0-second speech clip (below 2.0s forensic minimum duration)."""
    file_path = temp_dir / "short_speech.wav"
    sr = 16000
    t = np.linspace(0, 1.0, int(sr * 1.0))
    y = np.sin(2 * np.pi * 200 * t)
    sf.write(file_path, y, sr)
    return str(file_path)


@pytest.fixture
def silent_wav(temp_dir):
    """Generates a 3.0-second pure silence WAV file."""
    file_path = temp_dir / "silence.wav"
    sr = 16000
    y = np.zeros(int(sr * 3.0))
    sf.write(file_path, y, sr)
    return str(file_path)


# 1. Preprocessing Tests
def test_audio_preprocessing_real(real_speech_wav):
    config = PAudioConfig()
    preprocessed = load_and_preprocess_audio(real_speech_wav, config)
    assert preprocessed.usable is True
    assert preprocessed.sr == 16000
    assert preprocessed.speech_duration_sec >= 2.0
    assert preprocessed.audio is not None
    assert preprocessed.reason == "OK"


def test_audio_preprocessing_short(short_speech_wav):
    config = PAudioConfig()
    preprocessed = load_and_preprocess_audio(short_speech_wav, config)
    assert preprocessed.usable is False
    assert "below minimum required duration" in preprocessed.reason


def test_audio_preprocessing_silence(silent_wav):
    config = PAudioConfig()
    preprocessed = load_and_preprocess_audio(silent_wav, config)
    assert preprocessed.usable is False
    assert "No active speech detected" in preprocessed.reason


def test_audio_preprocessing_missing_file():
    config = PAudioConfig()
    preprocessed = load_and_preprocess_audio("non_existent_file.wav", config)
    assert preprocessed.usable is False
    assert "File not found" in preprocessed.reason


# 2. Feature Extraction & Robustness Tests
def test_feature_extraction_exact_dimensionality():
    sr = 16000
    t = np.linspace(0, 2.5, int(sr * 2.5))
    y = np.sin(2 * np.pi * 220 * t)
    features = extract_all_audio_features(y, sr)
    assert len(features.feature_vector) == 92, f"Expected 92 features, got {len(features.feature_vector)}"
    assert not np.isnan(features.feature_vector).any()
    assert not np.isinf(features.feature_vector).any()


def test_feature_extraction_nan_safety():
    sr = 16000
    y = np.zeros(sr * 2)  # Zero array test for division by zero / NaNs
    features = extract_all_audio_features(y, sr)
    assert len(features.feature_vector) == 92
    assert not np.isnan(features.feature_vector).any()
    assert not np.isinf(features.feature_vector).any()


# 3. F1 One-Class SVM Tests
def test_f1_one_class_svm_trained_only_on_real(temp_dir):
    model_path = temp_dir / "oc_svm.joblib"

    # Train only on real speech feature matrix
    X_real = np.random.randn(30, 92) + 5.0  # Real speech feature distribution
    detector = OneClassSVMDetector(nu=0.1)
    detector.fit(X_real)
    assert detector.is_fitted is True

    # Real sample inside distribution -> consistency score high
    raw_real, cons_real = detector.predict_score(X_real[0])
    assert cons_real >= 0.45

    # Synthetic sample outside distribution -> negative raw score & low consistency
    X_synthetic = np.random.randn(1, 92) - 20.0
    raw_synth, cons_synth = detector.predict_score(X_synthetic[0])
    assert raw_synth < 0.0
    assert cons_synth < 0.30

    detector.save(model_path)
    assert model_path.exists()


# 4. Pretrained Detector Adapter Tests
def test_pretrained_anti_spoof_detector_fallback():
    detector = PretrainedAntiSpoofDetector(model_name="invalid_non_existent_model_xyz")
    sr = 16000
    y = np.random.randn(sr * 2)
    score, details = detector.predict_similarity(y, sr)
    assert 0.0 <= score <= 1.0
    assert details["mode"] in ("offline_acoustic_representation", "fine_tuned_anti_spoof_neural")


# 5. F2 Replay Detector Tests
def test_f2_replay_detection(replayed_wav):
    config = PAudioConfig()
    preprocessed = load_and_preprocess_audio(replayed_wav, config)
    features = extract_all_audio_features(preprocessed.audio, preprocessed.sr)

    detector = F2ReplayDetector(config)
    result = detector.analyze(preprocessed, features)

    assert "pathway" in result
    assert "reverb_detected" in result
    assert "confidence" in result
    assert "evidence" in result
    assert result["pathway"] in ("REPLAYED_RECORDING", "DIRECT_GENUINE", "DIRECT_SYNTHETIC", "INCONCLUSIVE")
    assert isinstance(result["reverb_detected"], bool)


# 6. Master P-Audio Pipeline End-to-End JSON Schema Tests
def test_p_audio_pipeline_end_to_end_json_contract(real_speech_wav, temp_dir):
    config = PAudioConfig()
    pipeline = PAudioPipeline(config)

    output_json = temp_dir / "result.json"
    result = pipeline.analyze_clip(real_speech_wav, str(output_json))

    # Verify JSON Schema Contract
    assert result["status"] == "SUCCESS"
    assert "f1_zero_day" in result
    assert "f2_replay" in result
    assert "verdict" in result
    assert "confidence" in result
    assert "evidence" in result

    assert result["f1_zero_day"]["known_generator_similarity"] in ("LOW", "MED", "HIGH", "INCONCLUSIVE")
    assert result["f1_zero_day"]["natural_speech_consistency"] in ("LOW", "MED", "HIGH", "INCONCLUSIVE")
    assert result["f1_zero_day"]["verdict"] in ("REAL", "KNOWN_SYNTHETIC", "UNKNOWN_SYNTHETIC", "INCONCLUSIVE")

    assert result["f2_replay"]["pathway"] in ("DIRECT_GENUINE", "DIRECT_SYNTHETIC", "REPLAYED_RECORDING", "INCONCLUSIVE")
    assert isinstance(result["f2_replay"]["reverb_detected"], bool)

    assert output_json.exists()
    with open(output_json, "r") as f:
        loaded_json = json.load(f)
    assert loaded_json["clip_id"] == "real_speech"
