import os
import sys
import subprocess
import numpy as np
import cv2
import soundfile as sf
from pathlib import Path

def create_face_frame(width=640, height=480, mismatch=False, frame_idx=0):
    """Draw a face with eyes and irises for MediaPipe FaceMesh."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light grey background

    # Face oval
    cv2.ellipse(img, (width // 2, height // 2), (120, 160), 0, 0, 360, (180, 150, 130), -1)

    # Eyes
    left_eye_center = (width // 2 - 45, height // 2 - 30)
    right_eye_center = (width // 2 + 45, height // 2 - 30)

    # Sclera (white)
    cv2.ellipse(img, left_eye_center, (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, right_eye_center, (20, 12), 0, 0, 360, (255, 255, 255), -1)

    # Irises (dark blue)
    cv2.circle(img, left_eye_center, 8, (120, 40, 20), -1)
    cv2.circle(img, right_eye_center, 8, (120, 40, 20), -1)

    # Catchlights (bright spot)
    # If mismatch is True, put catchlight in different relative spots in left vs right eye
    lx, ly = left_eye_center
    rx, ry = right_eye_center
    if not mismatch:
        cv2.circle(img, (lx - 2, ly - 2), 3, (255, 255, 255), -1)
        cv2.circle(img, (rx - 2, ry - 2), 3, (255, 255, 255), -1)
    else:
        # Left eye light top-left, right eye light bottom-right (mismatch)
        cv2.circle(img, (lx - 4, ly - 4), 4, (255, 255, 255), -1)
        cv2.circle(img, (rx + 4, ry + 4), 4, (255, 255, 255), -1)

    # Nose & Mouth
    cv2.line(img, (width // 2, height // 2 - 10), (width // 2, height // 2 + 20), (140, 100, 80), 2)
    cv2.ellipse(img, (width // 2, height // 2 + 50), (30, 10), 0, 0, 180, (80, 40, 140), 2)

    return img

def make_video(output_mp4, audio_wav, duration_sec=3.0, fps=30, mismatch=False):
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_video = output_path.with_suffix(".temp.mp4")

    width, height = 640, 480
    num_frames = int(duration_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = create_face_frame(width, height, mismatch=mismatch, frame_idx=i)
        out.write(frame)

    out.release()

    # Mux audio and video with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_video),
        "-i", str(audio_wav),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    if temp_video.exists():
        temp_video.unlink()

    print(f"Generated: {output_path}")

def main():
    seed_dir = Path("data/seed_clips")
    seed_dir.mkdir(parents=True, exist_ok=True)

    real_audio = "data/demo/real_audio.wav"
    gen_audio = "data/demo/generated_audio.wav"

    # 1. Clip 01: Real video + Real audio
    make_video(seed_dir / "clip_01_real.mp4", real_audio, duration_sec=3.0, mismatch=False)

    # 2. Clip 02: AI Synthesis (Mismatch eyes + Synthetic audio)
    make_video(seed_dir / "clip_02_ai_synth.mp4", gen_audio, duration_sec=3.0, mismatch=True)

    # 3. Clip 03: Re-encoded / Conventional Editing (Re-encoded clip 01 video + Real audio)
    make_video(seed_dir / "clip_03_reencoded.mp4", real_audio, duration_sec=3.0, mismatch=False)

    # 4. Clip 04: Replayed Audio (Replayed audio + Real video)
    # Create simple reverberated audio for replay
    y, sr = sf.read(real_audio)
    # Add simple echo/reverb
    reverb_y = y.copy()
    delay = int(sr * 0.05) # 50ms delay
    reverb_y[delay:] += 0.5 * y[:-delay]
    reverb_wav = seed_dir / "reverb_audio.wav"
    sf.write(str(reverb_wav), reverb_y, sr)

    make_video(seed_dir / "clip_04_replayed.mp4", str(reverb_wav), duration_sec=3.0, mismatch=False)

if __name__ == "__main__":
    main()
