"""
P-Fusion engine.

Implements the core fusion pipeline:

    video JSON + audio JSON
        → load raw feature sections
        → extract F1/F2/F3/F4 dicts
        → normalise → F5Evidence
        → attribute → F5AttributionResult
        → build cause tree
        → return final JSON-serialisable result dict

Integration rules
-----------------
- Integrates ONLY through JSON:  never imports p_audio or p_video code.
- Missing detector outputs are handled safely (treated as UNKNOWN).
- Attribution logic stays entirely in attribution.py.
- Cause-tree logic stays entirely in cause_tree.py.
- Output is a plain dict of JSON-serialisable Python types; see build_result().
- No new detection, thresholds, or fields are invented here.

Schema gap note
---------------
schemas.py defines F5FusionResult with a FinalVerdict field
(REAL | MANIPULATED | UNKNOWN).  Assigning FinalVerdict would require a
confirmed rule for translating an attribution cause into a binary verdict —
no such rule is defined in the implementation plan or docs/contracts.md.
Therefore fusion_engine returns a plain dict rather than an F5FusionResult
dataclass, avoiding the need to invent a FinalVerdict computation.

CLI
---
    uv run python -m p_fusion.fusion_engine \\
        --video-json path/to/video.json \\
        --audio-json path/to/audio.json

Both arguments are optional; omitting either treats the corresponding
detectors as absent (evidence for that feature set defaults to UNKNOWN).

No additional dependencies.
No imports from p_audio or p_video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from p_fusion.attribution import attribute
from p_fusion.cause_tree import build_cause_tree
from p_fusion.evidence_normalizer import normalize_evidence
from p_fusion.utils import load_json


# ---------------------------------------------------------------------------
# Feature extraction
#
# The raw JSON files produced by p_video and p_audio contain named sections
# for each feature.  These functions extract the appropriate sub-dict (or
# None when the section is absent) without touching any detector internals.
#
# Key names match docs/contracts.md exactly.
# ---------------------------------------------------------------------------

# Top-level section keys expected in each JSON envelope.
# These are the envelope-level keys, not internal detector field names.
_VIDEO_F3_KEY = "f3"   # Source/Propagation section in video JSON
_VIDEO_F4_KEY = "f4"   # Eye Reflection section in video JSON
_AUDIO_F1_KEY = "f1"   # Zero-Day Audio section in audio JSON
_AUDIO_F2_KEY = "f2"   # Replay Detection section in audio JSON


def _extract_f1(audio_data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the F1 (Zero-Day Audio) sub-dict from the audio envelope, or None."""
    if not isinstance(audio_data, dict):
        return None
    value = audio_data.get(_AUDIO_F1_KEY)
    return value if isinstance(value, dict) else None


def _extract_f2(audio_data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the F2 (Replay Detection) sub-dict from the audio envelope, or None."""
    if not isinstance(audio_data, dict):
        return None
    value = audio_data.get(_AUDIO_F2_KEY)
    return value if isinstance(value, dict) else None


def _extract_f3(video_data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the F3 (Source/Propagation) sub-dict from the video envelope, or None."""
    if not isinstance(video_data, dict):
        return None
    value = video_data.get(_VIDEO_F3_KEY)
    return value if isinstance(value, dict) else None


def _extract_f4(video_data: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return the F4 (Eye Reflection) sub-dict from the video envelope, or None."""
    if not isinstance(video_data, dict):
        return None
    value = video_data.get(_VIDEO_F4_KEY)
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Evidence serialiser
#
# Converts an F5Evidence dataclass into a plain JSON-serialisable dict,
# mirroring the field names from docs/contracts.md.  Enum members are
# emitted as their string value; None stays None.
# ---------------------------------------------------------------------------

def _serialise_evidence(ev) -> dict:
    """Return a JSON-serialisable representation of the F5Evidence."""
    meta = None
    if ev.f3.metadata is not None:
        meta = {
            "encoder": ev.f3.metadata.encoder,
            "creation_time": ev.f3.metadata.creation_time,
            "resolution": ev.f3.metadata.resolution,
        }

    return {
        "f1": {
            "known_generator_similarity": ev.f1.known_generator_similarity.value,
            "natural_speech_consistency": ev.f1.natural_speech_consistency.value,
            "verdict": ev.f1.verdict.value,
            "confidence": ev.f1.confidence,
        },
        "f2": {
            "pathway": ev.f2.pathway.value,
            "reverb_detected": ev.f2.reverb_detected,
            "confidence": ev.f2.confidence,
        },
        "f3": {
            "match_found": ev.f3.match_found,
            "matched_seed_id": ev.f3.matched_seed_id,
            "estimated_reencoding_stages": ev.f3.estimated_reencoding_stages,
            "metadata": meta,
        },
        "f4": {
            "frames_with_usable_eyes": ev.f4.frames_with_usable_eyes,
            "catchlight_mismatch_flagged_frames": list(ev.f4.catchlight_mismatch_flagged_frames),
            "mismatch_rate": ev.f4.mismatch_rate,
            "verdict": ev.f4.verdict.value,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_fusion(
    video_data: Optional[dict[str, Any]] = None,
    audio_data: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Execute the full P-Fusion pipeline on pre-loaded JSON dicts.

    This is the core function.  It accepts the already-parsed video and audio
    JSON envelopes (or None for either), runs the complete pipeline, and
    returns a single JSON-serialisable result dict.

    Pipeline
    --------
    1. Extract F1/F2 from audio_data; F3/F4 from video_data.
    2. Normalise → F5Evidence  (via evidence_normalizer.normalize_evidence).
    3. Attribute → F5AttributionResult  (via attribution.attribute).
    4. Build cause tree → dict  (via cause_tree.build_cause_tree).
    5. Assemble and return the final result dict.

    Args:
        video_data: Parsed video JSON envelope, or None.
        audio_data: Parsed audio JSON envelope, or None.

    Returns:
        A plain dict with the following top-level keys:

        "attribution":    str    — AttributionCause value
        "reason":         str    — human-readable explanation
        "has_conflict":   bool   — True if signals materially disagree
        "evidence":       dict   — normalised F1–F4 fields (contracts.md names)
        "cause_tree":     dict   — structured explanation from cause_tree.py
        "limitations":    list   — global limitations from the cause tree

    The dict is JSON-serialisable with json.dumps() without further
    transformation (no custom encoder needed).
    """
    # Step 1: extract feature sub-dicts from envelopes
    f1_data = _extract_f1(audio_data)
    f2_data = _extract_f2(audio_data)
    f3_data = _extract_f3(video_data)
    f4_data = _extract_f4(video_data)

    # Step 2: normalise → F5Evidence
    evidence = normalize_evidence(
        f1_data=f1_data,
        f2_data=f2_data,
        f3_data=f3_data,
        f4_data=f4_data,
    )

    # Step 3: attribute
    attribution_result = attribute(evidence)

    # Step 4: build cause tree
    tree = build_cause_tree(evidence, attribution_result)

    # Step 5: assemble result
    return {
        "attribution": attribution_result.cause.value,
        "reason": attribution_result.conflicts.details,
        "has_conflict": attribution_result.conflicts.has_conflict,
        "evidence": _serialise_evidence(evidence),
        "cause_tree": tree,
        "limitations": tree["limitations"],
    }


def run_fusion_from_files(
    video_json_path: Optional[Path | str] = None,
    audio_json_path: Optional[Path | str] = None,
) -> dict:
    """
    Load JSON files from disk and execute the fusion pipeline.

    Either path may be None or omitted; the corresponding detectors will be
    treated as absent and their evidence will default to UNKNOWN.

    Args:
        video_json_path: Path to the video results JSON file, or None.
        audio_json_path: Path to the audio results JSON file, or None.

    Returns:
        JSON-serialisable result dict from run_fusion().

    Raises:
        FileNotFoundError: If a supplied path does not exist.
        ValueError:        If a supplied file is not valid JSON or not a dict.
    """
    video_data: Optional[dict[str, Any]] = None
    audio_data: Optional[dict[str, Any]] = None

    if video_json_path is not None:
        video_data = load_json(video_json_path)

    if audio_json_path is not None:
        audio_data = load_json(audio_json_path)

    return run_fusion(video_data=video_data, audio_data=audio_data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p_fusion.fusion_engine",
        description=(
            "P-Fusion: fuse video and audio detector JSON outputs into a "
            "structured attribution result."
        ),
    )
    parser.add_argument(
        "--video-json",
        metavar="PATH",
        default=None,
        help="Path to the video detector results JSON file (optional).",
    )
    parser.add_argument(
        "--audio-json",
        metavar="PATH",
        default=None,
        help="Path to the audio detector results JSON file (optional).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=False,
        help="Pretty-print the JSON output (default: compact).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point.

    Prints the fusion result as JSON to stdout.
    Returns 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_fusion_from_files(
            video_json_path=args.video_json,
            audio_json_path=args.audio_json,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
