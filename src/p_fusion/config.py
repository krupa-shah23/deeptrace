"""
P-Fusion configuration.

Contains only constants and path conventions that P-Fusion itself requires.
No detector thresholds. No attribution rules. No F1/F2/F3/F4 internals.

Sources of truth:
- docs/contracts.md  — field names, allowed values, missing-data semantics
- README.md          — directory layout conventions
- schemas.py         — internal typed representations
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file's location.
# src/p_fusion/config.py → up two levels → repo root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Output directory layout
# Matches the conventions documented in README.md
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
VIDEO_RESULTS_DIR: Path = OUTPUT_DIR / "video_results"
AUDIO_RESULTS_DIR: Path = OUTPUT_DIR / "audio_results"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"

# ---------------------------------------------------------------------------
# Input contract keys
# Exact field names from docs/contracts.md — do not rename.
# ---------------------------------------------------------------------------

# F1 keys (audio_results JSON)
F1_KEY_KNOWN_GENERATOR_SIMILARITY = "known_generator_similarity"
F1_KEY_NATURAL_SPEECH_CONSISTENCY = "natural_speech_consistency"
F1_KEY_VERDICT = "verdict"
F1_KEY_CONFIDENCE = "confidence"

# F2 keys (audio_results JSON)
F2_KEY_PATHWAY = "pathway"
F2_KEY_REVERB_DETECTED = "reverb_detected"
F2_KEY_CONFIDENCE = "confidence"

# F3 keys (video_results JSON)
F3_KEY_MATCH_FOUND = "match_found"
F3_KEY_MATCHED_SEED_ID = "matched_seed_id"
F3_KEY_ESTIMATED_REENCODING_STAGES = "estimated_reencoding_stages"
F3_KEY_METADATA = "metadata"
F3_KEY_METADATA_ENCODER = "encoder"
F3_KEY_METADATA_CREATION_TIME = "creation_time"
F3_KEY_METADATA_RESOLUTION = "resolution"

# F4 keys (video_results JSON)
F4_KEY_FRAMES_WITH_USABLE_EYES = "frames_with_usable_eyes"
F4_KEY_CATCHLIGHT_MISMATCH_FLAGGED_FRAMES = "catchlight_mismatch_flagged_frames"
F4_KEY_MISMATCH_RATE = "mismatch_rate"
F4_KEY_VERDICT = "verdict"

# ---------------------------------------------------------------------------
# Semantic assumptions — explicitly required by the integration specification.
# These are semantic facts from contracts.md, not invented thresholds.
# ---------------------------------------------------------------------------

# F3: a no-match is absence of evidence, not evidence of manipulation.
# Stated explicitly in contracts.md: "match_found=false means NO MATCH,
# not proof of manipulation."
F3_NO_MATCH_IS_NOT_PROOF_OF_MANIPULATION: bool = True

# F4: INSUFFICIENT_DATA must remain distinguishable from CONSISTENT.
# Stated explicitly in contracts.md.
# Downstream code must never coerce INSUFFICIENT_DATA → CONSISTENT.
F4_INSUFFICIENT_DATA_DISTINCT_FROM_CONSISTENT: bool = True

# F3: estimated_reencoding_stages is a heuristic, not an exact count.
# Stated explicitly in contracts.md.
F3_REENCODING_STAGES_IS_HEURISTIC: bool = True

# ---------------------------------------------------------------------------
# Report output configuration (placeholders — no logic yet)
# reportlab / jinja2 / weasyprint are listed in pyproject.toml.
# ---------------------------------------------------------------------------

REPORT_TITLE: str = "DeepTrace Forensic Report"
REPORT_FILE_SUFFIX: str = "_report.pdf"
