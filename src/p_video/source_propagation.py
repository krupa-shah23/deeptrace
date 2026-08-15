import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
import json
import subprocess
import numpy as np
import cv2
import imagehash
from PIL import Image
from preprocessing import extract_frames

# ---------------------------------------------------------------------------
# Step 1 — perceptual hash fingerprint per video
# ---------------------------------------------------------------------------

def video_fingerprint(video_path, sample_every_n=15):
    """Returns a list of imagehash.ImageHash objects, one per sampled frame."""
    hashes = []
    for idx, ts, frame in extract_frames(video_path, sample_every_n=sample_every_n):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hashes.append(imagehash.phash(pil_img))
    return hashes


# ---------------------------------------------------------------------------
# Step 2 — build the DB (seed clips + their known ffmpeg-degraded variants)
# ---------------------------------------------------------------------------

def build_database(seed_paths, sample_every_n=15):
    """
    seed_paths: dict of {video_id: path}, e.g.
        {"seed_003": "data/seed_clips/video.mp4"}
    Returns {video_id: [ImageHash, ...]}
    """
    db = {}
    for video_id, path in seed_paths.items():
        print(f"[db] fingerprinting {video_id} ({path})")
        db[video_id] = video_fingerprint(path, sample_every_n=sample_every_n)
    return db


# ---------------------------------------------------------------------------
# Step 3 — match a suspect video against the DB
# ---------------------------------------------------------------------------

def match_against_db(suspect_hashes, db, threshold=10):
    """
    Returns (best_match_id_or_None, best_avg_hamming_distance).
    Compares position-by-position across the shorter of the two hash lists.
    """
    best_match, best_dist = None, float("inf")
    for video_id, db_hashes in db.items():
        n = min(len(suspect_hashes), len(db_hashes))
        if n == 0:
            continue
        dists = [suspect_hashes[i] - db_hashes[i] for i in range(n)]
        avg_dist = sum(dists) / n
        if avg_dist < best_dist:
            best_dist, best_match = avg_dist, video_id
    if best_dist <= threshold:
        return best_match, best_dist
    return None, best_dist


# ---------------------------------------------------------------------------
# Step 4 — metadata layer (ffprobe)
# ---------------------------------------------------------------------------

def get_metadata(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", video_path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )
    fmt = data.get("format", {})

    return {
        "encoder": fmt.get("tags", {}).get("encoder", "unknown"),
        "resolution": f"{video_stream.get('width', '?')}x{video_stream.get('height', '?')}",
        "codec": video_stream.get("codec_name", "unknown"),
        "creation_time": fmt.get("tags", {}).get("creation_time", "unknown"),
        "bit_rate": fmt.get("bit_rate", "unknown"),
    }


# ---------------------------------------------------------------------------
# Step 5 — estimate re-encoding stages
# ---------------------------------------------------------------------------

def estimate_reencoding_stages(suspect_meta, matched_meta):
    """
    Simple heuristic: count how many 'degradation dimensions' differ from
    the matched seed. Explicitly an estimate, not an exact reconstruction —
    documented as such in the report.
    """
    if matched_meta is None:
        return None
    stages = 0
    if suspect_meta.get("resolution") != matched_meta.get("resolution"):
        stages += 1
    if suspect_meta.get("codec") != matched_meta.get("codec"):
        stages += 1
    try:
        suspect_br = int(suspect_meta.get("bit_rate", 0))
        matched_br = int(matched_meta.get("bit_rate", 0))
        if matched_br > 0 and abs(suspect_br - matched_br) / matched_br > 0.25:
            stages += 1
    except (ValueError, TypeError):
        pass
    return stages


# ---------------------------------------------------------------------------
# Main — build DB, run a suspect video through it, emit P-Fusion's JSON contract
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suspect", required=True, help="path to suspect video")
    parser.add_argument("--seed", required=True, help="path to seed video (video_id=seed's filename stem)")
    parser.add_argument("--variants-dir", default="data/seed_clips/variants",
                         help="dir containing ffmpeg-degraded variants of the seed")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--out", default=None, help="optional path to write JSON output")
    args = parser.parse_args()

    # Build DB: seed + all its variants
    seed_id = "seed_" + os.path.splitext(os.path.basename(args.seed))[0]
    seed_paths = {seed_id: args.seed}

    if os.path.isdir(args.variants_dir):
        for fname in os.listdir(args.variants_dir):
            if fname.lower().endswith((".mp4", ".mov", ".mkv")):
                variant_id = "seed_" + os.path.splitext(fname)[0]
                seed_paths[variant_id] = os.path.join(args.variants_dir, fname)

    print(f"[db] building database from {len(seed_paths)} clips: {list(seed_paths.keys())}")
    db = build_database(seed_paths)

    # Fingerprint + match the suspect
    print(f"\n[match] fingerprinting suspect: {args.suspect}")
    suspect_hashes = video_fingerprint(args.suspect)
    matched_id, dist = match_against_db(suspect_hashes, db, threshold=args.threshold)

    print(f"[match] best match: {matched_id} (avg hamming distance: {dist:.2f}, threshold: {args.threshold})")

    # Metadata comparison
    suspect_meta = get_metadata(args.suspect)
    matched_meta = get_metadata(seed_paths[matched_id]) if matched_id else None
    stages = estimate_reencoding_stages(suspect_meta, matched_meta) if matched_id else None

    print(f"[metadata] suspect: {suspect_meta}")
    if matched_meta:
        print(f"[metadata] matched seed: {matched_meta}")
        print(f"[metadata] estimated re-encoding stages: {stages}")

    # Output contract — matches P-Fusion's agreed JSON shape exactly
    output = {
        "feature": "source_propagation",
        "match_found": matched_id is not None,
        "matched_seed_id": matched_id,
        "estimated_reencoding_stages": stages if stages is not None else 0,
        "metadata": suspect_meta,
    }

    print("\n--- JSON for P-Fusion ---")
    print(json.dumps(output, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWritten to {args.out}")