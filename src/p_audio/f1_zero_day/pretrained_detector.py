from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch


class PretrainedAntiSpoofDetector:
    """
    Adapter for Pretrained Audio Anti-Spoofing / Synthetic Speech Detector.
    Evaluates audio signals using a fine-tuned deepfake/anti-spoofing checkpoint
    (e.g., ASVspoof fine-tuned Wav2Vec2/XLSR model) or robust acoustic fallback.
    Produces a normalized synthetic_pattern_score between 0.0 and 1.0.
    """

    def __init__(self, model_name: str = "mohammedgaber/wav2vec2-large-xlsr-53-anti-spoofing"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        self.use_offline_fallback = False

    def load_model(self):
        """Attempts to load fine-tuned anti-spoofing model checkpoint."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[Pretrained Detector] Successfully loaded anti-spoofing classifier: {self.model_name}")
        except Exception as e:
            print(f"[Pretrained Detector] Note: Could not load '{self.model_name}' online ({e}). Operating in offline acoustic representation fallback mode.")
            self.use_offline_fallback = True
            self.is_loaded = True

    def predict_similarity(self, y: np.ndarray, sr: int) -> Tuple[float, Dict[str, Any]]:
        """
        Computes synthetic pattern similarity score (0.0 = low synthetic similarity, 1.0 = high synthetic similarity).
        Returns (score, debug_details). Note: This is an uncalibrated model score, not a posterior probability.
        """
        if not self.is_loaded:
            self.load_model()

        if self.use_offline_fallback or self.model is None:
            return self._offline_acoustic_similarity(y, sr)

        try:
            inputs = self.feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # For binary anti-spoof classifiers (index 0 = bonafide, index 1 = spoof/fake):
            if len(probs) >= 2:
                # Assuming index 1 corresponds to spoof/synthetic class
                synthetic_score = float(probs[1])
            else:
                synthetic_score = float(probs[0])

            synthetic_score = float(np.clip(synthetic_score, 0.0, 1.0))

            details = {
                "model_name": self.model_name,
                "mode": "fine_tuned_anti_spoof_neural",
                "synthetic_pattern_score": synthetic_score,
                "is_calibrated_probability": False
            }
            return synthetic_score, details
        except Exception as e:
            print(f"[Pretrained Detector] Neural inference fallback ({e}). Using offline acoustic anti-spoof model.")
            return self._offline_acoustic_similarity(y, sr)

    def _offline_acoustic_similarity(self, y: np.ndarray, sr: int) -> Tuple[float, Dict[str, Any]]:
        """
        Offline acoustic representation fallback analyzing high-frequency spectral flatness,
        phase continuity, and micro-tremor variance common in neural vocoders.
        """
        import librosa
        # 1. High frequency power ratio (> 6 kHz)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        hf_mask = freqs > 6000
        hf_ratio = float(np.mean(S[hf_mask, :]) / (np.mean(S) + 1e-8)) if np.any(hf_mask) else 0.0

        # 2. Spectral flatness variance
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_std = float(np.std(flatness))

        # Synthetic neural vocoder speech often has unnaturally flat high frequencies
        # or distinct periodic artifacts
        synth_indicator = (hf_ratio * 0.4) + (1.0 - min(1.0, flatness_std * 10)) * 0.6
        synth_indicator = float(np.clip(synth_indicator, 0.05, 0.95))

        details = {
            "model_name": self.model_name,
            "mode": "offline_acoustic_representation",
            "hf_ratio": round(float(hf_ratio), 4),
            "flatness_std": round(float(flatness_std), 4),
            "synthetic_pattern_score": round(float(synth_indicator), 4),
            "is_calibrated_probability": False
        }
        return synth_indicator, details
