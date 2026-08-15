import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
import numpy as np
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

def get_eye_crops(frame_bgr, iris_centers, crop_radius=25):
    """Given iris centers, return tight crops around each eye."""
    h, w = frame_bgr.shape[:2]
    crops = {}
    for eye_name, (cx, cy) in iris_centers.items():
        x1, y1 = max(0, int(cx - crop_radius)), max(0, int(cy - crop_radius))
        x2, y2 = min(w, int(cx + crop_radius)), min(h, int(cy + crop_radius))
        crops[eye_name] = frame_bgr[y1:y2, x1:x2]
    return crops


def find_catchlight(eye_crop_bgr, debug=False):
    if eye_crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    max_brightness = gray.max()

    if debug:
        print(f"    mean={mean_brightness:.1f}, max={max_brightness}")

    if mean_brightness < 40:
        return None
    if max_brightness < mean_brightness + 30:
        return None

    thresh_value = max(max_brightness - 15, mean_brightness + 30)
    _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(thresh > 0)
    if len(xs) == 0:
        return None

    cx, cy = xs.mean(), ys.mean()
    pixel_count = len(xs)  # use pixel count instead of contourArea — robust for tiny blobs
    return cx, cy, pixel_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to a test video")
    args = parser.parse_args()

    detected_count = 0
    catchlight_count = 0
    total_count = 0

    for idx, ts, frame in extract_frames(args.input, sample_every_n=5):
        total_count += 1
        iris_centers = get_iris_centers(frame)
        if not iris_centers:
            continue
        detected_count += 1

        eye_crops = get_eye_crops(frame, iris_centers)
        catchlights = {name: find_catchlight(crop, debug=(total_count <= 5)) for name, crop in eye_crops.items()}

        if total_count <= 5:  # print a few examples
            print(f"frame {idx} ({ts:.2f}s): iris={iris_centers}")
            print(f"    catchlights: {catchlights}")

        if all(c is not None for c in catchlights.values()):
            catchlight_count += 1

    print(f"\nDetected face+iris in {detected_count}/{total_count} sampled frames")
    print(f"Both catchlights found in {catchlight_count}/{detected_count} face-detected frames")