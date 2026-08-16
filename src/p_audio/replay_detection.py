import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path if executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from p_audio.config import PAudioConfig
from p_audio.f2_replay.detector import F2ReplayDetector
from p_audio.features import extract_all_audio_features
from p_audio.preprocess import load_and_preprocess_audio


def main():
    parser = argparse.ArgumentParser(description="P-Audio F2 Replay / Physical-World Re-recording Detection")
    parser.add_argument("--input", "-i", required=True, help="Path to input audio or video clip")
    parser.add_argument("--out", "-o", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    config = PAudioConfig()
    preprocessed = load_and_preprocess_audio(args.input, config)

    if not preprocessed.usable or preprocessed.audio is None:
        result = {
            "pathway": "INCONCLUSIVE",
            "reverb_detected": False,
            "confidence": 0.0,
            "reason": preprocessed.reason
        }
    else:
        features = extract_all_audio_features(preprocessed.audio, preprocessed.sr)
        detector = F2ReplayDetector(config)
        result = detector.analyze(preprocessed, features)

    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
