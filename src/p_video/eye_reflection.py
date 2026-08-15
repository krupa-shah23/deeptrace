import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))

import cv2
import mediapipe as mp
from preprocessing import extract_frames

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,  # gives iris landmarks
    max_num_faces=1,
    min_detection_confidence=0.5,
)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def get_iris_centers(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark

    def center_of(indices):
        xs = [lm[i].x * w for i in indices]
        ys = [lm[i].y * h for i in indices]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    return {"left_iris": center_of(LEFT_IRIS), "right_iris": center_of(RIGHT_IRIS)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to a test video")
    args = parser.parse_args()

    detected_count = 0
    total_count = 0
    for idx, ts, frame in extract_frames(args.input, sample_every_n=5):
        total_count += 1
        result = get_iris_centers(frame)
        if result:
            detected_count += 1
            if total_count <= 3:  # print a few examples
                print(f"frame {idx} ({ts:.2f}s): {result}")

    print(f"\nDetected face+iris in {detected_count}/{total_count} sampled frames")