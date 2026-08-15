import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import soundfile as sf

from src.p_audio.config import PAudioConfig
from src.p_audio.f1_zero_day.detector import F1ZeroDayDetector
from src.p_audio.f1_zero_day.model import OneClassSVMDetector
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
    """Generates a 3.0-second speech-like audio clip (real speech proxy with harmonics and pitch variation)."""
    file_path = temp_dir / "real_speech.wav"
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0))
    f0 = 140.0 + 10.0 * np.sin(2 * np.pi * 3.0 * t) + 2.0 * np.random.randn(len(t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = np.sin(phase) + 0.4 * np.sin(3.5 * phase)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    y = signal * envelope + 0.002 * np.random.randn(len(t))
    y = y / np.max(np.abs(y)) * 0.90
    sf.write(file_path, y, sr)
    return str(file_path)


@pytest.fixture
def short_speech_wav(temp_dir):
    """Generates a 1.0-second speech clip (too short for forensic threshold)."""
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


@pytest.fixture
def replayed_wav(temp_dir):
    """Generates a 3.0-second re-recorded audio clip with simulated reverberation (T60 decay) and elevated noise."""
    file_path = temp_dir / "replayed.wav"
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3.0))
    dry_speech = np.sin(2 * np.pi * 150 * t) * 0.5
    # Reverberation tail simulation
    reverb_impulse = np.exp(-5.0 * np.linspace(0, 0.5, int(sr * 0.5)))
    reverberant = np.convolve(dry_speech, reverb_impulse, mode="same")
    noise = 0.03 * np.random.randn(len(reverberant))  # elevated noise floor
    replayed_signal = reverberant + noise
    replayed_signal = replayed_signal / np.max(np.abs(replayed_signal)) * 0.90
    sf.write(file_path, replayed_signal, sr)
    return str(file_path)


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


def test_feature_extraction_nan_safety():
    sr = 16000
    y = np.zeros(sr * 2)  # Zero array test for division by zero / NaNs
    features = extract_all_audio_features(y, sr)
    assert not np.isnan(features.feature_vector).any()
    assert not np.isinf(features.feature_vector).any()
    assert len(features.feature_vector) > 30


def test_f1_one_class_svm_fit_save_load(temp_dir):
    model_path = temp_dir / "oc_svm.joblib"
    X_train = np.random.randn(20, 45)  # 20 samples, 45 features

    detector = OneClassSVMDetector(nu=0.1)
    detector.fit(X_train)
    assert detector.is_fitted is True

    raw_score, consistency = detector.predict_score(X_train[0])
    assert isinstance(raw_score, float)
    assert 0.0 <= consistency <= 1.0

    detector.save(model_path)
    assert model_path.exists()

    loaded_detector = OneClassSVMDetector.load(model_path)
    assert loaded_detector.is_fitted is True
    raw_score_2, consistency_2 = loaded_detector.predict_score(X_train[0])
    assert abs(raw_score - raw_score_2) < 1e-5


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
    assert isinstance(result["reverb_detected"], bool)


def test_p_audio_pipeline_end_to_end(real_speech_wav, temp_dir):
    config = PAudioConfig()
    pipeline = PAudioPipeline(config)

    output_json = temp_dir / "result.json"
    result = pipeline.analyze_clip(real_speech_wav, str(output_json))

    assert result["status"] == "SUCCESS"
    assert "f1_zero_day" in result
    assert "f2_replay" in result
    assert "verdict" in result
    assert "confidence" in result
    assert output_json.exists()

    with open(output_json, "r") as f:
        loaded_json = json.load(f)
    assert loaded_json["clip_id"] == "real_speech"
    assert loaded_json["verdict"] in ("REAL", "KNOWN_SYNTHETIC", "UNKNOWN_SYNTHETIC", "REPLAYED_RECORDING", "INCONCLUSIVE")
