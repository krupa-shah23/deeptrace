"""
P-Fusion cause tree builder.

Constructs a JSON-serialisable cause tree from a normalised F5Evidence and the
F5AttributionResult already produced by attribution.py.

Purpose
-------
The cause tree is a structured summary intended for forensic report rendering.
It explains *why* the attribution result was reached by annotating each evidence
branch with its role in the decision, preserving all raw feature values and
surfacing important semantic limitations.

Constraints (strictly enforced)
---------------------------------
- Does NOT perform new detection.
- Does NOT create new attribution rules or thresholds.
- Does NOT override or re-run attribution.py logic.
- Does NOT invent confidence scores or probabilities.
- F3 match_found=False is preserved as NO_MATCH, not manipulation evidence.
- UNKNOWN/None values are preserved as unknown — never inferred as positive
  or negative evidence.
- Output is a plain dict containing only JSON-serialisable Python types
  (str, bool, int, float, list, dict, None).

No imports from p_audio or p_video.
No additional dependencies beyond the standard library and p_fusion.schemas.
"""

from __future__ import annotations

from p_fusion.schemas import (
    AttributionCause,
    F1Verdict,
    F2Pathway,
    F4Verdict,
    F5AttributionResult,
    F5Evidence,
)

# ---------------------------------------------------------------------------
# Role labels
#
# CONTRIBUTING   — this feature's value was required by the attribution rule
#                  that fired; its signal is one of the listed rule conditions.
# NEUTRAL        — this feature produced a definitive value but was not part
#                  of the rule that fired (or is a partial signal under
#                  INCONCLUSIVE where no rule fired at all).
# UNKNOWN_DATA   — this feature produced no definitive data; its value is
#                  UNKNOWN or absent.  Neutral: it neither confirms nor denies
#                  any hypothesis.
# ---------------------------------------------------------------------------

_ROLE_CONTRIBUTING = "CONTRIBUTING"
_ROLE_NEUTRAL = "NEUTRAL"
_ROLE_UNKNOWN_DATA = "UNKNOWN_DATA"


# ---------------------------------------------------------------------------
# Field serialisers
#
# Each serialiser converts one F-evidence dataclass into a plain dict.
# Enum members are emitted as their string .value.
# None fields are kept as None (JSON null).
# ---------------------------------------------------------------------------

def _f1_fields(ev: F5Evidence) -> dict:
    return {
        "known_generator_similarity": ev.f1.known_generator_similarity.value,
        "natural_speech_consistency": ev.f1.natural_speech_consistency.value,
        "verdict": ev.f1.verdict.value,
        "confidence": ev.f1.confidence,
    }


def _f2_fields(ev: F5Evidence) -> dict:
    return {
        "pathway": ev.f2.pathway.value,
        "reverb_detected": ev.f2.reverb_detected,
        "confidence": ev.f2.confidence,
    }


def _f3_fields(ev: F5Evidence) -> dict:
    meta: dict | None = None
    if ev.f3.metadata is not None:
        meta = {
            "encoder": ev.f3.metadata.encoder,
            "creation_time": ev.f3.metadata.creation_time,
            "resolution": ev.f3.metadata.resolution,
        }
    return {
        "match_found": ev.f3.match_found,
        "matched_seed_id": ev.f3.matched_seed_id,
        "estimated_reencoding_stages": ev.f3.estimated_reencoding_stages,
        "metadata": meta,
    }


def _f4_fields(ev: F5Evidence) -> dict:
    return {
        "frames_with_usable_eyes": ev.f4.frames_with_usable_eyes,
        "catchlight_mismatch_flagged_frames": list(ev.f4.catchlight_mismatch_flagged_frames),
        "mismatch_rate": ev.f4.mismatch_rate,
        "verdict": ev.f4.verdict.value,
    }


# ---------------------------------------------------------------------------
# Role annotators
#
# Each annotator answers: "given the already-determined attribution cause,
# what role did this feature play?"
#
# Role is CONTRIBUTING only if:
#   (a) the cause matches a specific rule, AND
#   (b) this feature's value is exactly what that rule required.
#
# This is purely descriptive — the attribution decision is not re-made here.
# ---------------------------------------------------------------------------

def _f1_role(ev: F5Evidence, cause: AttributionCause) -> str:
    if ev.f1.verdict is F1Verdict.UNKNOWN:
        return _ROLE_UNKNOWN_DATA
    if cause is AttributionCause.AI_SYNTHESIS and ev.f1.verdict is F1Verdict.UNKNOWN_SYNTHETIC:
        return _ROLE_CONTRIBUTING
    if cause is AttributionCause.RE_RECORDED and ev.f1.verdict is F1Verdict.KNOWN_SYNTHETIC:
        return _ROLE_CONTRIBUTING
    if cause is AttributionCause.CONVENTIONAL_EDITING and ev.f1.verdict is F1Verdict.REAL:
        return _ROLE_CONTRIBUTING
    return _ROLE_NEUTRAL


def _f2_role(ev: F5Evidence, cause: AttributionCause) -> str:
    if ev.f2.pathway is F2Pathway.UNKNOWN:
        return _ROLE_UNKNOWN_DATA
    if cause is AttributionCause.RE_RECORDED and ev.f2.pathway is F2Pathway.REPLAYED_RECORDING:
        return _ROLE_CONTRIBUTING
    return _ROLE_NEUTRAL


def _f3_role(ev: F5Evidence, cause: AttributionCause) -> str:
    if ev.f3.match_found is None:
        return _ROLE_UNKNOWN_DATA
    # match_found=False contributes to AI_SYNTHESIS (with F4+F1) — not as manipulation
    if cause is AttributionCause.AI_SYNTHESIS and ev.f3.match_found is False:
        return _ROLE_CONTRIBUTING
    if cause is AttributionCause.CONVENTIONAL_EDITING and ev.f3.match_found is True:
        return _ROLE_CONTRIBUTING
    return _ROLE_NEUTRAL


def _f4_role(ev: F5Evidence, cause: AttributionCause) -> str:
    if ev.f4.verdict is F4Verdict.UNKNOWN:
        return _ROLE_UNKNOWN_DATA
    if cause is AttributionCause.AI_SYNTHESIS and ev.f4.verdict is F4Verdict.MISMATCH_DETECTED:
        return _ROLE_CONTRIBUTING
    if cause is AttributionCause.CONVENTIONAL_EDITING and ev.f4.verdict is F4Verdict.CONSISTENT:
        return _ROLE_CONTRIBUTING
    return _ROLE_NEUTRAL


# ---------------------------------------------------------------------------
# Per-branch semantic notes
#
# Notes surface critical semantic constraints that could be misread without
# context — especially the F3 NO_MATCH / manipulation distinction and the
# F4 UNKNOWN / CONSISTENT distinction.
# ---------------------------------------------------------------------------

def _f3_notes(ev: F5Evidence) -> list:
    notes = []
    if ev.f3.match_found is False:
        notes.append(
            "match_found=False means NO SOURCE MATCH was found in the seed "
            "database. This is absence of a known source, not evidence of "
            "manipulation. It does not independently indicate synthetic "
            "generation and only contributes to AI_SYNTHESIS attribution in "
            "combination with F4=MISMATCH_DETECTED and F1=UNKNOWN_SYNTHETIC."
        )
    if ev.f3.match_found is None:
        notes.append(
            "F3 data was absent. match_found is unknown. "
            "Unknown evidence is neutral."
        )
    return notes


def _f4_notes(ev: F5Evidence) -> list:
    notes = []
    if ev.f4.verdict is F4Verdict.UNKNOWN:
        notes.append(
            "F4 verdict is UNKNOWN. This may represent an INSUFFICIENT_DATA "
            "verdict from the detector (normalised to UNKNOWN) or absent F4 "
            "data. UNKNOWN cannot be interpreted as CONSISTENT — it is neutral."
        )
    return notes


# ---------------------------------------------------------------------------
# Global limitations
#
# Limitations are tree-level notes that apply to the entire result rather than
# to a specific evidence branch.
# ---------------------------------------------------------------------------

def _global_limitations(ev: F5Evidence, attribution: F5AttributionResult) -> list:
    limitations = [
        "This cause tree is explanatory only. It annotates the existing "
        "attribution result; it does not re-run detection or change the "
        "attribution decision."
    ]

    if attribution.cause is AttributionCause.UNKNOWN and not attribution.conflicts.has_conflict:
        limitations.append(
            "Attribution is INCONCLUSIVE. No specific cause was established. "
            "Evidence that was present is listed per branch but was insufficient "
            "alone or in combination to satisfy any attribution rule."
        )

    if attribution.conflicts.has_conflict:
        limitations.append(
            "One or more definitive detector signals materially disagree. "
            "A specific attribution cannot be made reliably under conflict. "
            "See 'reason' for the specific contradiction."
        )

    # Enumerate features that contributed no definitive data
    absent = []
    if ev.f1.verdict is F1Verdict.UNKNOWN:
        absent.append("F1 (Zero-Day Audio)")
    if ev.f2.pathway is F2Pathway.UNKNOWN:
        absent.append("F2 (Replay Detection)")
    if ev.f3.match_found is None:
        absent.append("F3 (Source/Propagation)")
    if ev.f4.verdict is F4Verdict.UNKNOWN:
        absent.append("F4 (Eye Reflection)")

    if absent:
        limitations.append(
            f"The following features reported no definitive data: "
            f"{', '.join(absent)}. "
            "Absent or unknown evidence is neutral — it neither confirms "
            "nor denies any attribution hypothesis."
        )

    return limitations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cause_tree(
    evidence: F5Evidence,
    attribution: F5AttributionResult,
) -> dict:
    """
    Build a JSON-serialisable cause tree from F5Evidence and an F5AttributionResult.

    The cause tree is a plain dict containing only JSON-serialisable Python
    types (str, bool, int, float, list, dict, None).  It is suitable for
    direct JSON serialisation or for use as input to a report renderer.

    Structure
    ---------
    {
        "attribution": str,           # AttributionCause value
        "reason": str,                # human-readable explanation from attribution.py
        "has_conflict": bool,         # True when signals materially disagree
        "evidence_branches": [        # one entry per F1–F4 feature
            {
                "feature": str,           # "F1" | "F2" | "F3" | "F4"
                "label": str,             # human-readable feature name
                "role": str,              # CONTRIBUTING | NEUTRAL | UNKNOWN_DATA
                "fields": dict,           # raw field values (enums as strings)
                "notes": list[str],       # semantic notes for this feature
            }
        ],
        "limitations": list[str],     # tree-level caveats and constraints
    }

    Role meanings
    -------------
    CONTRIBUTING  — feature's value satisfied one of the fired rule's conditions.
    NEUTRAL       — feature produced a definitive value not required by the rule,
                    or produced a partial signal under an INCONCLUSIVE result.
    UNKNOWN_DATA  — feature produced no definitive data (UNKNOWN/absent).

    Args:
        evidence:    Normalised F5Evidence from the evidence normalizer.
        attribution: F5AttributionResult from attribution.attribute().

    Returns:
        A plain dict that is JSON-serialisable without further transformation.
    """
    cause = attribution.cause

    branches = [
        {
            "feature": "F1",
            "label": "Zero-Day Audio",
            "role": _f1_role(evidence, cause),
            "fields": _f1_fields(evidence),
            "notes": [],
        },
        {
            "feature": "F2",
            "label": "Replay Detection",
            "role": _f2_role(evidence, cause),
            "fields": _f2_fields(evidence),
            "notes": [],
        },
        {
            "feature": "F3",
            "label": "Source/Propagation",
            "role": _f3_role(evidence, cause),
            "fields": _f3_fields(evidence),
            "notes": _f3_notes(evidence),
        },
        {
            "feature": "F4",
            "label": "Eye Reflection",
            "role": _f4_role(evidence, cause),
            "fields": _f4_fields(evidence),
            "notes": _f4_notes(evidence),
        },
    ]

    return {
        "attribution": cause.value,
        "reason": attribution.conflicts.details,
        "has_conflict": attribution.conflicts.has_conflict,
        "evidence_branches": branches,
        "limitations": _global_limitations(evidence, attribution),
    }
