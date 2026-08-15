import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
import numpy as np
import cv2
import mediapipe as mp
from preprocessing import extract_frames
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,  # gives iris landmarks
    max_num_faces=1,
    min_detection_confidence=0.5,
)
import json

def format_timestamp(ts_seconds):
    """Convert seconds -> HH:MM:SS.ms string for the flagged_frames output."""
    total_seconds = ts_seconds
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def get_face_geometry(frame_bgr):
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

    iris_centers = {
        "left_iris": center_of(LEFT_IRIS),
        "right_iris": center_of(RIGHT_IRIS),
    }

    lx = lm[LEFT_EYE_OUTER].x * w
    ly = lm[LEFT_EYE_OUTER].y * h
    rx = lm[RIGHT_EYE_OUTER].x * w
    ry = lm[RIGHT_EYE_OUTER].y * h

    roll = np.degrees(np.arctan2(ry - ly, rx - lx))

    return iris_centers, roll

class CatchlightSmoother:
    """Keeps a short history of raw (cx, cy) per eye and returns the median position."""
    def __init__(self, window=3):
        self.window = window
        self.history = {"left_iris": deque(maxlen=window), "right_iris": deque(maxlen=window)}

    def update(self, eye_name, catchlight):
        if catchlight is None:
            self.history[eye_name].clear()  # reset on gap — don't smooth across a dropped detection
            return None
        cx, cy, pixel_count = catchlight
        self.history[eye_name].append((cx, cy))
        pts = self.history[eye_name]
        med_cx = float(np.median([p[0] for p in pts]))
        med_cy = float(np.median([p[1] for p in pts]))
        return med_cx, med_cy, pixel_count

def get_iris_centers_and_roll(frame_bgr):
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

    iris_centers = {
        "left_iris": center_of(LEFT_IRIS),
        "right_iris": center_of(RIGHT_IRIS),
    }

    # Eye-corner line → approximate head roll
    lx = lm[LEFT_EYE_OUTER].x * w
    ly = lm[LEFT_EYE_OUTER].y * h

    rx = lm[RIGHT_EYE_OUTER].x * w
    ry = lm[RIGHT_EYE_OUTER].y * h

    roll = np.degrees(np.arctan2(ry - ly, rx - lx))

    return iris_centers, roll

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

def compute_offset_vector_normalized(catchlight, roll_degrees, crop_radius=25):
    cx, cy, _ = catchlight

    dx = cx - crop_radius
    dy = cy - crop_radius

    # Convert image coordinates to a vector.
    # Rotate the vector by -roll to compensate for head roll.
    theta = np.radians(-roll_degrees)

    dx_norm = dx * np.cos(theta) - dy * np.sin(theta)
    dy_norm = dx * np.sin(theta) + dy * np.cos(theta)

    magnitude = np.sqrt(dx_norm**2 + dy_norm**2) / crop_radius
    normalized_angle = np.degrees(np.arctan2(dy_norm, dx_norm))

    return normalized_angle, magnitude


def angle_to_bucket(angle_degrees, num_buckets=8):
    """Quantize into 8 directional bins (N, NE, E, SE, S, SW, W, NW)."""
    normalized = (angle_degrees + 360) % 360
    bucket_size = 360 / num_buckets
    return int((normalized + bucket_size / 2) // bucket_size) % num_buckets


def check_mismatch(
    left_catchlight,
    right_catchlight,
    roll_degrees,
    bucket_tolerance=1
):
    if left_catchlight is None or right_catchlight is None:
        return "INSUFFICIENT_DATA", None, None, None

    left_angle, left_mag = compute_offset_vector_normalized(
        left_catchlight,
        roll_degrees
    )

    right_angle, right_mag = compute_offset_vector_normalized(
        right_catchlight,
        roll_degrees
    )

    left_bucket = angle_to_bucket(left_angle)
    right_bucket = angle_to_bucket(right_angle)

    bucket_diff = min(
        abs(left_bucket - right_bucket),
        8 - abs(left_bucket - right_bucket)
    )

    verdict = (
        "MISMATCH"
        if bucket_diff > bucket_tolerance
        else "CONSISTENT"
    )

    angle_diff = abs(left_angle - right_angle)
    angle_diff = min(angle_diff, 360 - angle_diff)

    return verdict, angle_diff, left_angle, right_angle

def rotate_crop_for_roll(crop, roll_degrees):
    h, w = crop.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        roll_degrees,
        1.0
    )

    return cv2.warpAffine(
        crop,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def check_mismatch_baseline(angle_diff, baseline_diff, deviation_threshold=40):
    """
    Flags MISMATCH only when angle_diff deviates from this clip's own
    established baseline — not from zero. Baseline absorbs per-person/
    per-lighting systematic offset; deviation captures real anomalies.
    """
    if angle_diff is None or baseline_diff is None:
        return "INSUFFICIENT_DATA"
    deviation = abs(angle_diff - baseline_diff)
    deviation = min(deviation, 360 - deviation)
    return "MISMATCH" if deviation > deviation_threshold else "CONSISTENT"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to a test video")
    parser.add_argument("--out", default=None, help="optional path to write JSON output")
    args = parser.parse_args()

    detected_count = 0
    catchlight_count = 0
    total_count = 0
    mismatch_count = 0
    consistent_count = 0
    insufficient_count = 0

    angle_diffs = []
    flagged_frames = []

    smoother = CatchlightSmoother(window=3)

    for idx, ts, frame in extract_frames(args.input, sample_every_n=5):
        total_count += 1

        geometry = get_iris_centers_and_roll(frame)

        if geometry is None:
            continue

        iris_centers, roll = geometry

        detected_count += 1

        eye_crops = get_eye_crops(frame, iris_centers)

        catchlights = {
            name: find_catchlight(crop)
            for name, crop in eye_crops.items()
        }

        # Apply 3-frame median smoothing before comparing
        smoothed = {
            name: smoother.update(name, catchlights[name])
            for name in catchlights
        }

        if all(c is not None for c in smoothed.values()):
            catchlight_count += 1

        verdict, angle_diff, left_angle, right_angle = check_mismatch(
            smoothed["left_iris"],
            smoothed["right_iris"],
            roll
        )
        if angle_diff is not None:
            angle_diffs.append(angle_diff)

        if verdict == "MISMATCH":
            mismatch_count += 1
            flagged_frames.append(format_timestamp(ts))

        elif verdict == "CONSISTENT":
            consistent_count += 1

        else:
            insufficient_count += 1

        if total_count <= 10:
            if left_angle is not None and right_angle is not None:
                _, left_mag = compute_offset_vector_normalized(
                    smoothed["left_iris"], roll
                )
                _, right_mag = compute_offset_vector_normalized(
                    smoothed["right_iris"], roll
                )

                print(
                    f"frame {idx} ({ts:.2f}s): "
                    f"roll={roll:.1f}°, "
                    f"left(angle={left_angle:.1f}, mag={left_mag:.3f}), "
                    f"right(angle={right_angle:.1f}, mag={right_mag:.3f}), "
                    f"diff={angle_diff:.1f}"
                )
            else:
                print(
                    f"frame {idx} ({ts:.2f}s): "
                    f"INSUFFICIENT_DATA"
                )

    print(f"\nDetected face+iris in {detected_count}/{total_count} sampled frames")
    print(f"Both catchlights found in {catchlight_count}/{detected_count} face-detected frames")
    print(f"Verdicts — CONSISTENT: {consistent_count}, MISMATCH: {mismatch_count}, INSUFFICIENT_DATA: {insufficient_count}")
    

    if consistent_count + mismatch_count > 0:
        mismatch_rate = mismatch_count / (consistent_count + mismatch_count)
        print(
            f"Mismatch rate (excluding insufficient-data frames): "
            f"{mismatch_rate:.2%}"
        )

        if angle_diffs:
            print(
                f"angle_diff stats — "
                f"median: {np.median(angle_diffs):.1f}, "
                f"min: {min(angle_diffs):.1f}, "
                f"max: {max(angle_diffs):.1f}"
            )

            # Full distribution
            print(
                f"\nFull angle_diff distribution "
                f"(all {len(angle_diffs)} frames):"
            )
            print(
                sorted(round(d, 1) for d in angle_diffs)
            )

            print(
                f"25th/50th/75th percentile: "
                f"{np.percentile(angle_diffs, 25):.1f} / "
                f"{np.percentile(angle_diffs, 50):.1f} / "
                f"{np.percentile(angle_diffs, 75):.1f}"
            )

            # Establish baseline from first 15 valid frames
            if len(angle_diffs) >= 15:
                baseline_window = angle_diffs[:15]

                baseline_diff = float(
                    np.median(baseline_window)
                )

                # Median Absolute Deviation
                mad = float(
                    np.median(
                        [
                            abs(d - baseline_diff)
                            for d in baseline_window
                        ]
                    )
                )

                # Robust threshold
                deviation_threshold = max(
                    6 * mad,
                    25
                )

                print(
                    f"\nBaseline: {baseline_diff:.1f}, "
                    f"MAD: {mad:.1f}, "
                    f"threshold: {deviation_threshold:.1f}"
                )

                # Re-classify all frames using baseline + MAD threshold
                baseline_mismatch_count = sum(
                    1
                    for d in angle_diffs
                    if check_mismatch_baseline(
                        d,
                        baseline_diff,
                        deviation_threshold
                    ) == "MISMATCH"
                )

                print(
                    f"Baseline-relative mismatch count: "
                    f"{baseline_mismatch_count}/{len(angle_diffs)}"
                )

            else:
                print(
                    "\nCould not establish baseline: "
                    "fewer than 15 valid angle_diff values."
                )

            # ---- Build P-Fusion output contract ----
    total_usable = consistent_count + mismatch_count
    frames_with_usable_eyes_pct = round(
        100 * total_usable / detected_count, 1
    ) if detected_count > 0 else 0.0

    if total_usable == 0:
        verdict_out = "INSUFFICIENT_DATA"
    elif mismatch_count / total_usable > 0.5:
        # more than half the usable frames flagged -> report as MISMATCH_DETECTED
        verdict_out = "MISMATCH_DETECTED"
    else:
        verdict_out = "CONSISTENT"

    mismatch_rate_out = round(mismatch_count / total_usable, 4) if total_usable > 0 else 0.0

    output = {
        "feature": "eye_reflection_consistency",
        "frames_with_usable_eyes": frames_with_usable_eyes_pct,
        "catchlight_mismatch_flagged_frames": flagged_frames,
        "mismatch_rate": mismatch_rate_out,
        "verdict": verdict_out,
    }

    print("\n--- JSON for P-Fusion ---")
    print(json.dumps(output, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWritten to {args.out}")