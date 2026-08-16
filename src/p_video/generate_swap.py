"""
Generate a face-swapped version of a video using insightface's inswapper model
directly — no roop, no Colab, no Gradio. Processes frame-by-frame with OpenCV.

Usage:
    uv run python src/p_video/generate_swap.py \
        --source path/to/source_face.jpg \
        --target data/seed_clips/video.mp4 \
        --out data/seed_clips/video_swapped.mp4
"""
import argparse
import os
import cv2
import insightface
from insightface.app import FaceAnalysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path to a single source face image")
    parser.add_argument("--target", required=True, help="path to the target video to swap into")
    parser.add_argument("--out", required=True, help="path to write the swapped output video")
    args = parser.parse_args()

    print("[setup] loading face analysis model...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("[setup] loading inswapper model...")
    inswapper_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
    if not os.path.exists(inswapper_path):
        raise FileNotFoundError(
            f"inswapper_128.onnx not found at {inswapper_path} — "
            "confirm it was moved into ~/.insightface/models/"
        )
    swapper = insightface.model_zoo.get_model(
        inswapper_path, download=False, download_zip=False
    )

    print(f"[source] reading source face from {args.source}")
    source_img = cv2.imread(args.source)
    source_faces = face_app.get(source_img)
    if not source_faces:
        raise RuntimeError("No face detected in source image — try a clearer, more frontal photo.")
    source_face = source_faces[0]

    print(f"[target] opening video {args.target}")
    cap = cv2.VideoCapture(args.target)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[target] {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0
    swapped_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target_faces = face_app.get(frame)
        if target_faces:
            frame = swapper.get(frame, target_faces[0], source_face, paste_back=True)
            swapped_count += 1

        out.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"\nDone. {swapped_count}/{frame_idx} frames had a detected face and were swapped.")
    print(f"Output written to {args.out}")

if __name__ == "__main__":
    main()