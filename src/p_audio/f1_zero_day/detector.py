from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

try:
    from p_audio.config import PAudioConfig
    from p_audio.f1_zero_day.model import OneClassSVMDetector
    from p_audio.f1_zero_day.pretrained_detector import PretrainedAntiSpoofDetector
    from p_audio.features import AudioFeatures
    from p_audio.preprocess import PreprocessedAudio
except ImportError:
    from p_audio.config import PAudioConfig
    from p_audio.f1_zero_day.model import OneClassSVMDetector
    from p_audio.f1_zero_day.pretrained_detector import PretrainedAntiSpoofDetector
    from p_audio.features import AudioFeatures
    from p_audio.preprocess import PreprocessedAudio


class F1ZeroDayDetector:
    """
    F1 — Zero-Day / Unknown-Generator Synthetic Audio Detector.
    Combines:
    1. Pretrained Anti-Spoofing Detector (known generator similarity / synthetic pattern score)
    2. One-Class SVM trained strictly on genuine human speech (natural speech consistency score)
    """

    def __init__(self, config: Optional[PAudioConfig] = None):
        self.config = config or PAudioConfig()
        self.pretrained_detector = PretrainedAntiSpoofDetector(
            model_name=self.config.pretrained_model_name
        )
        self.oc_svm_detector: Optional[OneClassSVMDetector] = None
        self._initialize_models()

    def _initialize_models(self):
        """Loads One-Class SVM model artifact if it exists, or initializes default model."""
        model_path = Path(self.config.f1_model_path)
        if model_path.exists():
            try:
                self.oc_svm_detector = OneClassSVMDetector.load(model_path)
                print(f"[F1 Detector] Loaded One-Class SVM model from {model_path}")
            except Exception as e:
                print(f"[F1 Detector] Failed to load One-Class SVM model ({e}). Will use dynamic baseline.")
                self.oc_svm_detector = None
        else:
            print(f"[F1 Detector] No pre-trained One-Class SVM found at {model_path}. Run train_f1.py to generate artifact.")
            self.oc_svm_detector = None

    def analyze(self, preprocessed: PreprocessedAudio, features: AudioFeatures) -> Dict[str, Any]:
        """
        Executes F1 Zero-Day analysis on preprocessed audio clip.
        Returns structured JSON result matching P-Fusion interface contract.
        Note: Output 'confidence' is a normalized score between 0.0 and 1.0, not a calibrated Bayesian probability.
        """
        if not preprocessed.usable or preprocessed.audio is None:
            return {
                "known_generator_similarity": "INCONCLUSIVE",
                "natural_speech_consistency": "INCONCLUSIVE",
                "verdict": "INCONCLUSIVE",
                "confidence": 0.0,
                "reason": preprocessed.reason
            }

        # 1. Pretrained Anti-Spoofing Model Score / Synthetic Pattern Score
        sim_score, pretrained_details = self.pretrained_detector.predict_similarity(
            preprocessed.audio, preprocessed.sr
        )

        if sim_score >= self.config.f1_similarity_threshold_high:
            known_gen_sim = "HIGH"
        elif sim_score >= self.config.f1_similarity_threshold_med:
            known_gen_sim = "MED"
        else:
            known_gen_sim = "LOW"

        # 2. One-Class SVM Natural-Speech Consistency Score
        if self.oc_svm_detector and self.oc_svm_detector.is_fitted:
            raw_oc_score, consistency_score = self.oc_svm_detector.predict_score(
                features.feature_vector
            )
        else:
            # Fallback heuristic using micro-tremor & spectral features when model artifact is missing
            jitter = features.details.get("jitter_local", 0.0)
            shimmer = features.details.get("shimmer_local", 0.0)
            f0_cv = features.details.get("f0_cv", 0.0)

            # Natural speech exhibits non-zero micro-tremor and normal pitch variation
            if jitter > 0.001 and shimmer > 0.01 and f0_cv > 0.02:
                consistency_score = 0.80
                raw_oc_score = 0.5
            else:
                consistency_score = 0.20
                raw_oc_score = -0.5

        if consistency_score >= 0.50:
            nat_speech_cons = "HIGH"
        elif consistency_score >= 0.35:
            nat_speech_cons = "MED"
        else:
            nat_speech_cons = "LOW"

        # 3. Fusion Logic for Zero-Day Classification
        if known_gen_sim == "HIGH":
            verdict = "KNOWN_SYNTHETIC"
            confidence = round(float(sim_score), 4)
        elif known_gen_sim in ("LOW", "MED") and nat_speech_cons == "LOW":
            # Anomaly detected outside real speech distribution without matching a known generator -> UNKNOWN_SYNTHETIC
            verdict = "UNKNOWN_SYNTHETIC"
            confidence = round(float(1.0 - consistency_score), 4)
        elif nat_speech_cons in ("HIGH", "MED") and known_gen_sim == "LOW":
            verdict = "REAL"
            confidence = round(float(consistency_score), 4)
        else:
            verdict = "INCONCLUSIVE"
            confidence = 0.50

        return {
            "known_generator_similarity": known_gen_sim,
            "natural_speech_consistency": nat_speech_cons,
            "verdict": verdict,
            "confidence": confidence,
            "details": {
                "synthetic_pattern_score_raw": round(float(sim_score), 4),
                "consistency_score_raw": round(float(consistency_score), 4),
                "oc_svm_raw_score": round(float(raw_oc_score), 4),
                "is_calibrated_probability": False,
                "pretrained_model": pretrained_details
            }
        }
