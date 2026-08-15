import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.p_audio.config import PAudioConfig
from src.p_audio.f1_zero_day.detector import F1ZeroDayDetector
from src.p_audio.f2_replay.detector import F2ReplayDetector
from src.p_audio.features import extract_all_audio_features
from src.p_audio.preprocess import load_and_preprocess_audio


class PAudioPipeline:
    """
    Master P-Audio Forensic Pipeline for DeepTrace.
    Integrates F1 (Zero-Day Synthetic Audio Detection) & F2 (Replay / Re-recording Detection).
    Generates machine-readable structured JSON outputs for P-Fusion.
    """

    def __init__(self, config: Optional[PAudioConfig] = None):
        self.config = config or PAudioConfig()
        self.f1_detector = F1ZeroDayDetector(self.config)
        self.f2_detector = F2ReplayDetector(self.config)

    def analyze_clip(self, file_path: str, output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes end-to-end P-Audio analysis on input audio or video file.
        Returns combined JSON dictionary and optionally saves to output file.
        """
        start_time = time.time()
        file_path_obj = Path(file_path)

        # 1. Preprocessing
        preprocessed = load_and_preprocess_audio(str(file_path_obj), self.config)

        if not preprocessed.usable or preprocessed.audio is None:
            output = {
                "clip_id": file_path_obj.stem,
                "status": "INCONCLUSIVE",
                "reason": preprocessed.reason,
                "f1_zero_day": {
                    "known_generator_similarity": "INCONCLUSIVE",
                    "natural_speech_consistency": "INCONCLUSIVE",
                    "verdict": "INCONCLUSIVE",
                    "confidence": 0.0
                },
                "f2_replay": {
                    "pathway": "INCONCLUSIVE",
                    "reverb_detected": False,
                    "confidence": 0.0
                },
                "verdict": "INCONCLUSIVE",
                "confidence": 0.0,
                "evidence": {
                    "reason": preprocessed.reason,
                    "duration_sec": preprocessed.duration_sec,
                    "speech_duration_sec": preprocessed.speech_duration_sec
                }
            }
            if output_json_path:
                self._save_json(output, output_json_path)
            return output

        # 2. Feature Extraction
        features = extract_all_audio_features(preprocessed.audio, preprocessed.sr)

        # 3. F1 Zero-Day Detection
        f1_result = self.f1_detector.analyze(preprocessed, features)

        # 4. F2 Replay Detection
        f2_result = self.f2_detector.analyze(preprocessed, features, f1_verdict=f1_result.get("verdict"))

        processing_time_sec = round(time.time() - start_time, 4)

        # 5. Combined Verdict Determination for P-Fusion
        # Order of precedence:
        # 1. Replayed Recording detected by F2
        # 2. Known Synthetic detected by F1
        # 3. Unknown Synthetic detected by F1
        # 4. Real / Direct Genuine
        if f2_result.get("pathway") == "REPLAYED_RECORDING" and f2_result.get("confidence", 0.0) > 0.60:
            top_verdict = "REPLAYED_RECORDING"
            top_confidence = f2_result.get("confidence", 0.0)
        elif f1_result.get("verdict") == "KNOWN_SYNTHETIC":
            top_verdict = "KNOWN_SYNTHETIC"
            top_confidence = f1_result.get("confidence", 0.0)
        elif f1_result.get("verdict") == "UNKNOWN_SYNTHETIC":
            top_verdict = "UNKNOWN_SYNTHETIC"
            top_confidence = f1_result.get("confidence", 0.0)
        elif f1_result.get("verdict") == "REAL":
            top_verdict = "REAL"
            top_confidence = f1_result.get("confidence", 0.0)
        else:
            top_verdict = "INCONCLUSIVE"
            top_confidence = 0.0

        output = {
            "clip_id": file_path_obj.stem,
            "status": "SUCCESS",
            "f1_zero_day": {
                "known_generator_similarity": f1_result.get("known_generator_similarity", "INCONCLUSIVE"),
                "natural_speech_consistency": f1_result.get("natural_speech_consistency", "INCONCLUSIVE"),
                "verdict": f1_result.get("verdict", "INCONCLUSIVE"),
                "confidence": f1_result.get("confidence", 0.0)
            },
            "f2_replay": {
                "pathway": f2_result.get("pathway", "INCONCLUSIVE"),
                "reverb_detected": f2_result.get("reverb_detected", False),
                "confidence": f2_result.get("confidence", 0.0)
            },
            # Top-level direct fields for P-Fusion backwards compatibility
            "known_generator_similarity": f1_result.get("known_generator_similarity", "INCONCLUSIVE"),
            "natural_speech_consistency": f1_result.get("natural_speech_consistency", "INCONCLUSIVE"),
            "pathway": f2_result.get("pathway", "INCONCLUSIVE"),
            "reverb_detected": f2_result.get("reverb_detected", False),
            "verdict": top_verdict,
            "confidence": top_confidence,
            "evidence": {
                "preprocessing": {
                    "total_duration_sec": round(preprocessed.duration_sec, 2),
                    "speech_duration_sec": round(preprocessed.speech_duration_sec, 2),
                    "sampling_rate": preprocessed.sr
                },
                "f1_details": f1_result.get("details", {}),
                "f2_evidence": f2_result.get("evidence", {}),
                "feature_summary": features.details,
                "processing_time_sec": processing_time_sec
            }
        }

        if output_json_path:
            self._save_json(output, output_json_path)

        return output

    def _save_json(self, data: Dict[str, Any], path_str: str):
        out_path = Path(path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[P-Audio] Saved forensic result to {out_path}")
