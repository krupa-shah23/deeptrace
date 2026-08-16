import cv2

def extract_frames(video_path, sample_every_n=5):
    """Yields (frame_index, timestamp_sec, frame_bgr) for every Nth frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError(f"Could not read FPS for {video_path} — check the file is valid")
    print(f"[preprocessing] {video_path}: {fps:.2f} fps")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every_n == 0:
            yield idx, idx / fps, frame
        idx += 1
    cap.release()