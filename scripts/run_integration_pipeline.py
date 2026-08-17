import os
import sys
import json
import subprocess
from pathlib import Path

CLIPS = [
    ("clip_01_real", "data/seed_clips/clip_01_real.mp4", "data/seed_clips/clip_01_real.mp4"),
    ("clip_02_ai_synth", "data/seed_clips/clip_02_ai_synth.mp4", "data/seed_clips/clip_01_real.mp4"),
    ("clip_03_reencoded", "data/seed_clips/clip_03_reencoded.mp4", "data/seed_clips/clip_01_real.mp4"),
    ("clip_04_replayed", "data/seed_clips/clip_04_replayed.mp4", "data/seed_clips/clip_01_real.mp4"),
]

def run_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res

def main():
    out_audio_dir = Path("output/audio_results")
    out_video_dir = Path("output/video_results")
    out_combined_dir = Path("output/combined")
    out_f5_dir = Path("output/f5_results")

    for d in [out_audio_dir, out_video_dir, out_combined_dir, out_f5_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results_summary = []

    for clip_id, clip_path, seed_path in CLIPS:
        print(f"\n==========================================")
        print(f" Processing Clip: {clip_id}")
        print(f"==========================================")

        wav_path = Path("output") / f"{clip_id}.wav"

        # Step 5a: Extract Audio
        cmd_extract = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-vn", "-ac", "1", "-ar", "16000",
            str(wav_path)
        ]
        res_ext = run_cmd(cmd_extract)
        if res_ext.returncode != 0:
            print(f"[ERROR] Audio extraction failed for {clip_path}: {res_ext.stderr}")

        # Step 5b: Run F1 & F2
        f1_json_path = out_audio_dir / f"{clip_id}_f1.json"
        f2_json_path = out_audio_dir / f"{clip_id}_f2.json"

        cmd_f1 = ["uv", "run", "python", "src/p_audio/zero_day_detection.py", "--input", str(wav_path), "--out", str(f1_json_path)]
        res_f1 = run_cmd(cmd_f1)
        print(f"F1 return code: {res_f1.returncode}")
        if res_f1.returncode != 0:
            print(f"F1 stderr: {res_f1.stderr}")

        cmd_f2 = ["uv", "run", "python", "src/p_audio/replay_detection.py", "--input", str(wav_path), "--out", str(f2_json_path)]
        res_f2 = run_cmd(cmd_f2)
        print(f"F2 return code: {res_f2.returncode}")
        if res_f2.returncode != 0:
            print(f"F2 stderr: {res_f2.stderr}")

        # Step 5c: Run F3 & F4
        f3_json_path = out_video_dir / f"{clip_id}_f3.json"
        f4_json_path = out_video_dir / f"{clip_id}_f4.json"

        cmd_f3 = ["uv", "run", "python", "src/p_video/source_propagation.py", "--suspect", clip_path, "--seed", seed_path, "--out", str(f3_json_path)]
        res_f3 = run_cmd(cmd_f3)
        print(f"F3 return code: {res_f3.returncode}")
        if res_f3.returncode != 0:
            print(f"F3 stderr: {res_f3.stderr}")

        cmd_f4 = ["uv", "run", "python", "src/p_video/eye_reflection.py", "--input", clip_path, "--out", str(f4_json_path)]
        res_f4 = run_cmd(cmd_f4)
        print(f"F4 return code: {res_f4.returncode}")
        if res_f4.returncode != 0:
            print(f"F4 stderr: {res_f4.stderr}")

        # Load raw outputs
        f1_raw = json.loads(f1_json_path.read_text()) if f1_json_path.exists() else None
        f2_raw = json.loads(f2_json_path.read_text()) if f2_json_path.exists() else None
        f3_raw = json.loads(f3_json_path.read_text()) if f3_json_path.exists() else None
        f4_raw = json.loads(f4_json_path.read_text()) if f4_json_path.exists() else None

        # Step 5e: Combine envelopes
        audio_combined_path = out_combined_dir / f"{clip_id}_audio.json"
        video_combined_path = out_combined_dir / f"{clip_id}_video.json"

        audio_combined = {"f1": f1_raw, "f2": f2_raw}
        video_combined = {"f3": f3_raw, "f4": f4_raw}

        audio_combined_path.write_text(json.dumps(audio_combined, indent=2))
        video_combined_path.write_text(json.dumps(video_combined, indent=2))

        # Step 6: Run F5 Fusion Engine
        f5_result_path = out_f5_dir / f"{clip_id}_result.json"
        cmd_f5 = [
            "uv", "run", "python", "src/p_fusion/fusion_engine.py",
            "--video-json", str(video_combined_path),
            "--audio-json", str(audio_combined_path),
            "--pretty"
        ]
        res_f5 = run_cmd(cmd_f5)
        print(f"F5 return code: {res_f5.returncode}")
        if res_f5.returncode == 0:
            f5_result_path.write_text(res_f5.stdout)
            f5_data = json.loads(res_f5.stdout)
            print(f"F5 Attribution: {f5_data.get('attribution')}")
        else:
            print(f"F5 stderr: {res_f5.stderr}")

if __name__ == "__main__":
    main()
