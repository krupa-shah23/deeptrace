from typing import Dict, Any, Optional

try:
    from p_audio.config import PAudioConfig
    from p_audio.features import AudioFeatures
    from p_audio.preprocess import PreprocessedAudio
except ImportError:
    from src.p_audio.config import PAudioConfig
    from src.p_audio.features import AudioFeatures
    from src.p_audio.preprocess import PreprocessedAudio


class F2ReplayDetector:
    """
    F2 — Replay / Physical-World Re-recording Detector.
    Analyzes acoustic evidence of physical playback (speaker -> room -> mic path):
    - Room reverberation / T60 energy decay time proxy (EDC Schroeder integration)
    - Elevated noise floor & noise consistency
    - Comb filtering spectral ripples from room reflection
    - High frequency response attenuation
    """

    def __init__(self, config: Optional[PAudioConfig] = None):
        self.config = config or PAudioConfig()

    def analyze(
        self,
        preprocessed: PreprocessedAudio,
        features: AudioFeatures,
        f1_verdict: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes F2 Replay analysis on preprocessed audio clip.
        Returns structured JSON result matching P-Fusion interface contract.
        All thresholds are prototype thresholds and marked PROTOTYPE / CALIBRATION REQUIRED.
        """
        if not preprocessed.usable or preprocessed.audio is None:
            return {
                "pathway": "INCONCLUSIVE",
                "reverb_detected": False,
                "confidence": 0.0,
                "reason": preprocessed.reason,
                "evidence": {}
            }

        details = features.details
        t60_proxy = details.get("t60_est_sec", 0.05)
        noise_floor_db = details.get("noise_floor_db", -60.0)
        snr_db = details.get("snr_db", 30.0)
        comb_index = details.get("comb_filter_index", 0.0)
        spectral_flatness = details.get("spectral_flatness", 0.0)

        # Reverb detection: T60 proxy > prototype threshold (0.3s) or high comb filtering ripple
        reverb_detected = (t60_proxy >= self.config.f2_reverb_t60_threshold_sec) or (comb_index >= self.config.f2_comb_filter_threshold)

        # Calculate replay score based on acoustic proxy indicators
        reverb_score = min(1.0, max(0.0, (t60_proxy - 0.15) / 0.40))
        noise_score = min(1.0, max(0.0, (noise_floor_db - self.config.f2_noise_floor_threshold_db) / 20.0))
        comb_score = min(1.0, max(0.0, comb_index / 0.50))

        replay_indicator = (reverb_score * 0.45) + (noise_score * 0.30) + (comb_score * 0.25)
        replay_indicator = float(min(0.99, max(0.01, replay_indicator)))

        # Determine pathway
        if replay_indicator >= 0.55 or (reverb_detected and noise_floor_db > -40.0):
            pathway = "REPLAYED_RECORDING"
            confidence = round(float(replay_indicator), 4)
        else:
            # Direct recording (no physical room playback detected)
            if f1_verdict in ("KNOWN_SYNTHETIC", "UNKNOWN_SYNTHETIC"):
                pathway = "DIRECT_SYNTHETIC"
            elif f1_verdict == "REAL":
                pathway = "DIRECT_GENUINE"
            else:
                pathway = "DIRECT_GENUINE"
            confidence = round(float(1.0 - replay_indicator), 4)

        evidence = {
            "t60_reverb_proxy_sec": round(float(t60_proxy), 4),
            "t60_estimated_sec": round(float(t60_proxy), 4),  # backwards compatibility
            "t60_threshold_sec": self.config.f2_reverb_t60_threshold_sec,
            "noise_floor_db": round(float(noise_floor_db), 2),
            "snr_db": round(float(snr_db), 2),
            "comb_filter_index": round(float(comb_index), 4),
            "spectral_flatness": round(float(spectral_flatness), 4),
            "replay_score": round(float(replay_indicator), 4),
            "threshold_status": "PROTOTYPE / CALIBRATION REQUIRED"
        }

        return {
            "pathway": pathway,
            "reverb_detected": bool(reverb_detected),
            "confidence": confidence,
            "evidence": evidence
        }
