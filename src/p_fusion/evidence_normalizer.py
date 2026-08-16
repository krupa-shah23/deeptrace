"""
P-Fusion evidence normalizer.

Validates and converts raw F1/F2/F3/F4 JSON dictionaries into the typed
F5Evidence container defined in schemas.py.

Pipeline:
    F1 JSON + F2 JSON + F3 JSON + F4 JSON
        → per-detector validation
        → per-detector normalisation
        → F5Evidence

Sources of truth:
    docs/contracts.md  — field names, allowed values, missing-data behaviour
    schemas.py         — internal typed representations
    config.py          — canonical field-name string constants
    utils.py           — generic JSON/dict helpers

No attribution logic.
No invented thresholds.
No imports from p_audio or p_video.
"""

from __future__ import annotations

from typing import Any, Optional

from p_fusion.config import (
    F1_KEY_CONFIDENCE,
    F1_KEY_KNOWN_GENERATOR_SIMILARITY,
    F1_KEY_NATURAL_SPEECH_CONSISTENCY,
    F1_KEY_VERDICT,
    F2_KEY_CONFIDENCE,
    F2_KEY_PATHWAY,
    F2_KEY_REVERB_DETECTED,
    F3_KEY_ESTIMATED_REENCODING_STAGES,
    F3_KEY_MATCH_FOUND,
    F3_KEY_MATCHED_SEED_ID,
    F3_KEY_METADATA,
    F3_KEY_METADATA_CREATION_TIME,
    F3_KEY_METADATA_ENCODER,
    F3_KEY_METADATA_RESOLUTION,
    F4_KEY_CATCHLIGHT_MISMATCH_FLAGGED_FRAMES,
    F4_KEY_FRAMES_WITH_USABLE_EYES,
    F4_KEY_MISMATCH_RATE,
    F4_KEY_VERDICT,
)
from p_fusion.schemas import (
    F1Evidence,
    F1Verdict,
    F2Evidence,
    F2Pathway,
    F3Evidence,
    F3Metadata,
    F4Evidence,
    F4Verdict,
    F5Evidence,
    Level,
)


# ---------------------------------------------------------------------------
# Private field-level parsers
# ---------------------------------------------------------------------------

def _parse_level(raw: Any) -> Level:
    """Map a raw JSON string to Level. Returns Level.UNKNOWN on any failure."""
    if not isinstance(raw, str):
        return Level.UNKNOWN
    try:
        return Level(raw)
    except ValueError:
        return Level.UNKNOWN


def _parse_f1_verdict(raw: Any) -> F1Verdict:
    """Map a raw JSON string to F1Verdict. Returns F1Verdict.UNKNOWN on failure."""
    if not isinstance(raw, str):
        return F1Verdict.UNKNOWN
    try:
        return F1Verdict(raw)
    except ValueError:
        return F1Verdict.UNKNOWN


def _parse_f2_pathway(raw: Any) -> F2Pathway:
    """Map a raw JSON string to F2Pathway. Returns F2Pathway.UNKNOWN on failure."""
    if not isinstance(raw, str):
        return F2Pathway.UNKNOWN
    try:
        return F2Pathway(raw)
    except ValueError:
        return F2Pathway.UNKNOWN


def _parse_confidence(raw: Any) -> Optional[float]:
    """
    Parse a 0.0–1.0 float field (confidence or mismatch_rate).

    Returns None if the value is absent, non-numeric, or outside [0.0, 1.0].
    Does not raise; caller receives None and treats it as UNKNOWN.
    """
    if raw is None:
        return None
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return None
    value = float(raw)
    if not (0.0 <= value <= 1.0):
        return None
    return value


def _parse_bool(raw: Any) -> Optional[bool]:
    """Parse a JSON boolean field. Returns None for non-boolean values."""
    if isinstance(raw, bool):
        return raw
    return None


def _parse_nonneg_int(raw: Any) -> Optional[int]:
    """
    Parse a JSON integer field that must be non-negative.

    Returns None for non-integer values, booleans, or negative numbers.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    if raw < 0:
        return None
    return raw


def _parse_optional_str(raw: Any) -> Optional[str]:
    """Return the value unchanged if it is a str, else None."""
    return raw if isinstance(raw, str) else None


# ---------------------------------------------------------------------------
# Per-detector normalizers
# ---------------------------------------------------------------------------

def _normalize_f1(data: Optional[dict[str, Any]]) -> F1Evidence:
    """
    Validate and normalize an F1 (Zero-Day Audio) evidence dict.

    Contract fields consumed (docs/contracts.md §F1):
        known_generator_similarity  → Level (LOW | MED | HIGH)
        natural_speech_consistency  → Level (LOW | MED | HIGH)
        verdict                     → F1Verdict (REAL | KNOWN_SYNTHETIC | UNKNOWN_SYNTHETIC)
        confidence                  → float 0.0–1.0

    Normalisation rules:
        - Absent detector output (data is None) → all fields UNKNOWN / None.
        - Individual missing or unrecognised field value → that field UNKNOWN / None.
        - No field is silently renamed or fabricated.
    """
    if data is None:
        return F1Evidence()

    return F1Evidence(
        known_generator_similarity=_parse_level(
            data.get(F1_KEY_KNOWN_GENERATOR_SIMILARITY)
        ),
        natural_speech_consistency=_parse_level(
            data.get(F1_KEY_NATURAL_SPEECH_CONSISTENCY)
        ),
        verdict=_parse_f1_verdict(data.get(F1_KEY_VERDICT)),
        confidence=_parse_confidence(data.get(F1_KEY_CONFIDENCE)),
    )


def _normalize_f2(data: Optional[dict[str, Any]]) -> F2Evidence:
    """
    Validate and normalize an F2 (Replay Detection) evidence dict.

    Contract fields consumed (docs/contracts.md §F2):
        pathway          → F2Pathway (DIRECT_GENUINE | DIRECT_SYNTHETIC | REPLAYED_RECORDING)
        reverb_detected  → bool
        confidence       → float 0.0–1.0

    Normalisation rules:
        - Absent detector output (data is None) → all fields UNKNOWN / None.
        - Individual missing or unrecognised field value → that field UNKNOWN / None.
    """
    if data is None:
        return F2Evidence()

    return F2Evidence(
        pathway=_parse_f2_pathway(data.get(F2_KEY_PATHWAY)),
        reverb_detected=_parse_bool(data.get(F2_KEY_REVERB_DETECTED)),
        confidence=_parse_confidence(data.get(F2_KEY_CONFIDENCE)),
    )


def _normalize_f3(data: Optional[dict[str, Any]]) -> F3Evidence:
    """
    Validate and normalize an F3 (Source/Propagation) evidence dict.

    Contract fields consumed (docs/contracts.md §F3):
        match_found                  → bool
        matched_seed_id              → str | null
        estimated_reencoding_stages  → int (heuristic, 0-N)
        metadata.encoder             → str
        metadata.creation_time       → str
        metadata.resolution          → str

    Normalisation rules:
        - Absent detector output (data is None) → all fields None.
        - match_found=False is stored as-is.  It means NO MATCH and must
          never be interpreted as evidence of manipulation.
        - estimated_reencoding_stages is a heuristic; stored verbatim.
        - metadata sub-fields are each independently optional.
    """
    if data is None:
        return F3Evidence()

    raw_meta = data.get(F3_KEY_METADATA)
    metadata: Optional[F3Metadata] = None
    if isinstance(raw_meta, dict):
        metadata = F3Metadata(
            encoder=_parse_optional_str(raw_meta.get(F3_KEY_METADATA_ENCODER)),
            creation_time=_parse_optional_str(raw_meta.get(F3_KEY_METADATA_CREATION_TIME)),
            resolution=_parse_optional_str(raw_meta.get(F3_KEY_METADATA_RESOLUTION)),
        )

    return F3Evidence(
        match_found=_parse_bool(data.get(F3_KEY_MATCH_FOUND)),
        matched_seed_id=_parse_optional_str(data.get(F3_KEY_MATCHED_SEED_ID)),
        estimated_reencoding_stages=_parse_nonneg_int(
            data.get(F3_KEY_ESTIMATED_REENCODING_STAGES)
        ),
        metadata=metadata,
    )


def _normalize_f4(data: Optional[dict[str, Any]]) -> F4Evidence:
    """
    Validate and normalize an F4 (Eye Reflection) evidence dict.

    Contract fields consumed (docs/contracts.md §F4):
        frames_with_usable_eyes               → int ≥ 0
        catchlight_mismatch_flagged_frames     → list[str] of timestamps
        mismatch_rate                          → float 0.0–1.0
        verdict                               → CONSISTENT | MISMATCH_DETECTED | INSUFFICIENT_DATA

    Normalisation rules:
        - Absent detector output (data is None) → verdict UNKNOWN, numerics None.
        - INSUFFICIENT_DATA verdict → stored as UNKNOWN.
          Rationale: the detector could not gather enough usable frames; the
          result carries no positive evidence and is equivalent to absent data
          for attribution purposes.  mismatch_rate and frames_with_usable_eyes
          are still preserved verbatim so the report can document why the
          verdict was unavailable.
        - CONSISTENT and MISMATCH_DETECTED are passed through unchanged.
        - Non-string entries in catchlight_mismatch_flagged_frames are dropped.
    """
    if data is None:
        return F4Evidence()

    # Parse verdict; map INSUFFICIENT_DATA to UNKNOWN.
    verdict = F4Verdict.UNKNOWN
    raw_verdict = data.get(F4_KEY_VERDICT)
    if isinstance(raw_verdict, str):
        try:
            parsed_verdict = F4Verdict(raw_verdict)
            verdict = (
                F4Verdict.UNKNOWN
                if parsed_verdict is F4Verdict.INSUFFICIENT_DATA
                else parsed_verdict
            )
        except ValueError:
            verdict = F4Verdict.UNKNOWN

    # Parse flagged-frame timestamps; drop non-string entries silently.
    raw_flagged = data.get(F4_KEY_CATCHLIGHT_MISMATCH_FLAGGED_FRAMES)
    flagged_frames: list[str] = (
        [s for s in raw_flagged if isinstance(s, str)]
        if isinstance(raw_flagged, list)
        else []
    )

    return F4Evidence(
        frames_with_usable_eyes=_parse_nonneg_int(
            data.get(F4_KEY_FRAMES_WITH_USABLE_EYES)
        ),
        catchlight_mismatch_flagged_frames=flagged_frames,
        mismatch_rate=_parse_confidence(data.get(F4_KEY_MISMATCH_RATE)),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_evidence(
    f1_data: Optional[dict[str, Any]] = None,
    f2_data: Optional[dict[str, Any]] = None,
    f3_data: Optional[dict[str, Any]] = None,
    f4_data: Optional[dict[str, Any]] = None,
) -> F5Evidence:
    """
    Convert raw F1/F2/F3/F4 JSON dicts into a typed F5Evidence object.

    Each argument is an optional raw Python dict parsed from the corresponding
    detector's JSON output file.  Pass None for any detector whose output is
    unavailable — the corresponding evidence fields will be set to UNKNOWN.

    No attribution logic is applied here.  The returned F5Evidence is a pure,
    validated, contract-faithful representation of the raw detector outputs.

    Args:
        f1_data: F1 Zero-Day Audio detector output dict, or None.
        f2_data: F2 Replay Detection detector output dict, or None.
        f3_data: F3 Source/Propagation detector output dict, or None.
        f4_data: F4 Eye Reflection detector output dict, or None.

    Returns:
        F5Evidence — all sub-evidence objects are always present; missing or
        unrecognised values are represented as UNKNOWN / None, never silently
        promoted to positive evidence.
    """
    return F5Evidence(
        f1=_normalize_f1(f1_data),
        f2=_normalize_f2(f2_data),
        f3=_normalize_f3(f3_data),
        f4=_normalize_f4(f4_data),
    )
