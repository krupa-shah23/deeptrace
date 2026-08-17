import json
import subprocess
from pathlib import Path

def main():
    combined_dir = Path("output/combined")
    f5_dir = Path("output/f5_results")
    combined_dir.mkdir(parents=True, exist_ok=True)
    f5_dir.mkdir(parents=True, exist_ok=True)

    clips = [
        ("clip_a_real", "output/video_real_f1.json", "output/video_real_f2.json", "output/video_real_f3.json", "output/video_real_f4.json"),
        ("clip_b_swapped", "output/video_swapped_f1.json", "output/video_swapped_f2.json", "output/video_swapped_f3.json", "output/video_swapped_f4.json"),
        ("clip_c_reencoded", "output/video_reencoded_f1.json", "output/video_reencoded_f2.json", "output/video_reencoded_f3.json", "output/video_reencoded_f4.json"),
    ]

    for clip_id, f1_p, f2_p, f3_p, f4_p in clips:
        f1 = json.loads(Path(f1_p).read_text())
        f2 = json.loads(Path(f2_p).read_text())
        f3 = json.loads(Path(f3_p).read_text())
        f4 = json.loads(Path(f4_p).read_text())

        audio_env = {"f1": f1, "f2": f2}
        video_env = {"f3": f3, "f4": f4}

        audio_path = combined_dir / f"{clip_id}_audio.json"
        video_path = combined_dir / f"{clip_id}_video.json"
        f5_out_path = f5_dir / f"{clip_id}_result.json"

        audio_path.write_text(json.dumps(audio_env, indent=2))
        video_path.write_text(json.dumps(video_env, indent=2))

        cmd = [
            "uv", "run", "python", "src/p_fusion/fusion_engine.py",
            "--video-json", str(video_path),
            "--audio-json", str(audio_path),
            "--pretty"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"\n==========================================")
        print(f" Clip: {clip_id}")
        print(f" Return code: {res.returncode}")
        if res.returncode == 0:
            f5_out_path.write_text(res.stdout)
            data = json.loads(res.stdout)
            print(f" Attribution: {data.get('attribution')}")
            print(f" Reason: {data.get('reason')}")
            print(f" Has Conflict: {data.get('has_conflict')}")
        else:
            print(f" ERROR stderr: {res.stderr}")

if __name__ == "__main__":
    main()
