import argparse
import json
import sys
from pathlib import Path

# Add project root and src to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from p_audio.config import PAudioConfig
from p_audio.pipeline import PAudioPipeline


def main():
    parser = argparse.ArgumentParser(description="DeepTrace P-Audio Master Forensic Analyzer")
    parser.add_argument("input", help="Path to suspect audio file (WAV, MP3, FLAC, M4A) or video container (MP4, MKV)")
    parser.add_argument("--out", "-o", default=None, help="Output path for forensic JSON result (default: output/audio_results/<clip_id>.json)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist at '{args.input}'", file=sys.stderr)
        sys.exit(1)

    output_path = args.out
    if not output_path:
        out_dir = Path("output/audio_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{input_path.stem}.json")

    config = PAudioConfig()
    pipeline = PAudioPipeline(config)
    result = pipeline.analyze_clip(str(input_path), output_json_path=output_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
