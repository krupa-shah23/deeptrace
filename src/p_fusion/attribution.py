"""
P-Fusion attribution engine.

Applies deterministic, rule-based attribution to a normalised F5Evidence object,
producing an F5AttributionResult.

Sources:
    confirmed P-Fusion attribution rules
    src/p_fusion/schemas.py  — internal typed representations
    docs/contracts.md        — detector field semantics

Schema mapping (verified before implementation):
    AI_SYNTHESIS               → AttributionCause.AI_SYNTHESIS
    CONVENTIONAL_EDITING       → AttributionCause.CONVENTIONAL_EDITING
    AI_GENERATED_THEN_RECORDED → AttributionCause.RE_RECORDED
    INCONCLUSIVE               → AttributionCause.UNKNOWN
    CONFLICTING_EVIDENCE       → AttributionCause.UNKNOWN
                                 + AttributionConflict(has_conflict=True)
    Reasoning text             → AttributionConflict.details (str)

No numerical thresholds, weights, or probabilities.
No imports from p_audio or p_video.
"""

from __future__ import annotations

from p_fusion.schemas import (
    AttributionCause,
    AttributionConflict,
    F1Verdict,
    F2Pathway,
    F4Verdict,
    F5AttributionResult,
    F5Evidence,
)


# ---------------------------------------------------------------------------
# Evidence predicate helpers
#
# Each predicate tests exactly one definitive detector signal.
# UNKNOWN / None values always return False — they are never treated as
# positive or negative evidence.
# ---------------------------------------------------------------------------

def _f1_is_real(ev: F5Evidence) -> bool:
    return ev.f1.verdict is F1Verdict.REAL


def _f1_is_known_synthetic(ev: F5Evidence) -> bool:
    return ev.f1.verdict is F1Verdict.KNOWN_SYNTHETIC


def _f1_is_unknown_synthetic(ev: F5Evidence) -> bool:
    return ev.f1.verdict is F1Verdict.UNKNOWN_SYNTHETIC


def _f1_has_strong_synthetic(ev: F5Evidence) -> bool:
    """True when F1 produces any synthetic verdict (KNOWN_SYNTHETIC or UNKNOWN_SYNTHETIC)."""
    return ev.f1.verdict in (F1Verdict.KNOWN_SYNTHETIC, F1Verdict.UNKNOWN_SYNTHETIC)


def _f2_is_replayed(ev: F5Evidence) -> bool:
    return ev.f2.pathway is F2Pathway.REPLAYED_RECORDING


def _f3_match_found_true(ev: F5Evidence) -> bool:
    """True only when F3 explicitly produced match_found=True."""
    return ev.f3.match_found is True


def _f3_match_found_false(ev: F5Evidence) -> bool:
    """
    True only when F3 explicitly produced match_found=False.

    None (absent F3 data) is NOT treated as a no-match.
    match_found=False means NO MATCH — it is not manipulation evidence.
    """
    return ev.f3.match_found is False


def _f3_has_reencoding_data(ev: F5Evidence) -> bool:
    """
    True when F3 reported an estimated_reencoding_stages value.

    A value of 0 counts as data (direct copy from known source, no re-encoding).
    None means F3 data was absent.
    """
    return ev.f3.estimated_reencoding_stages is not None


def _f4_is_mismatch(ev: F5Evidence) -> bool:
    return ev.f4.verdict is F4Verdict.MISMATCH_DETECTED


def _f4_is_consistent(ev: F5Evidence) -> bool:
    return ev.f4.verdict is F4Verdict.CONSISTENT


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------

def _make_result(cause: AttributionCause, reason: str) -> F5AttributionResult:
    """Build a non-conflict F5AttributionResult, storing *reason* in details."""
    return F5AttributionResult(
        cause=cause,
        conflicts=AttributionConflict(has_conflict=False, details=reason),
    )


# ---------------------------------------------------------------------------
# Public attribution API
# ---------------------------------------------------------------------------

def attribute(evidence: F5Evidence) -> F5AttributionResult:
    """
    Apply deterministic attribution rules to the normalised F5Evidence.

    Evaluation order
    ----------------
    1. AI_SYNTHESIS
       F4=MISMATCH_DETECTED AND F3=match_found=False AND F1=UNKNOWN_SYNTHETIC.
       All three required signals must be definitive.

    2. RE_RECORDED  (AI_GENERATED_THEN_RECORDED)
       F2=REPLAYED_RECORDING AND F1=KNOWN_SYNTHETIC.
       Evaluated before CONVENTIONAL_EDITING because it is more specific.

    3. CONVENTIONAL_EDITING
       F3=match_found=True AND re-encoding data present
       AND explicitly non-synthetic evidence from F1 (REAL) and F4 (CONSISTENT).
       UNKNOWN/neutral evidence cannot satisfy this.

    4. INCONCLUSIVE
       Insufficient evidence to satisfy any specific attribution rule.

    Rule constraints (always upheld)
    ----------------------------------
    - F3 match_found=False is absence of a source match, NOT manipulation.
      It contributes to AI_SYNTHESIS only in combination with F4 and F1.
    - F4 is supporting evidence; MISMATCH_DETECTED alone does not determine cause.
    - UNKNOWN / None fields are neutral — they do not confirm or deny any rule.
    - No numerical thresholds, weights, or probabilities are used anywhere.

    Args:
        evidence: Normalised F5Evidence from the evidence normalizer.

    Returns:
        F5AttributionResult with cause and human-readable details explaining
        which signals drove the decision.
    """

    # ------------------------------------------------------------------
    # Step 1: AI_SYNTHESIS
    #
    # Required signals (all three must be definitive):
    #   F4 = MISMATCH_DETECTED  — video eye-reflection inconsistency detected
    #   F3 = match_found=False  — no known source seed (novel content)
    #   F1 = UNKNOWN_SYNTHETIC  — audio from an unrecognised synthetic generator
    # ------------------------------------------------------------------
    if (
        _f4_is_mismatch(evidence)
        and _f3_match_found_false(evidence)
        and _f1_is_unknown_synthetic(evidence)
    ):
        return _make_result(
            cause=AttributionCause.AI_SYNTHESIS,
            reason=(
                "AI_SYNTHESIS: all three required signals are present. "
                "F4=MISMATCH_DETECTED (eye-reflection catchlight inconsistency detected); "
                "F3=match_found=False (no known source seed — novel content); "
                "F1=UNKNOWN_SYNTHETIC (audio from an unrecognised synthetic generator). "
                "Note: F3 no-match alone is not manipulation evidence — it contributes "
                "here only in combination with F4 and F1."
            ),
        )

    # ------------------------------------------------------------------
    # Step 2: RE_RECORDED  (AI_GENERATED_THEN_RECORDED)
    #
    # Required signals:
    #   F2 = REPLAYED_RECORDING  — audio was physically re-recorded
    #   F1 = KNOWN_SYNTHETIC     — audio originates from a known TTS / AI system
    # ------------------------------------------------------------------
    if _f2_is_replayed(evidence) and _f1_is_known_synthetic(evidence):
        return _make_result(
            cause=AttributionCause.RE_RECORDED,
            reason=(
                "RE_RECORDED (AI_GENERATED_THEN_RECORDED): "
                "F2=REPLAYED_RECORDING (physical replay pathway detected); "
                "F1=KNOWN_SYNTHETIC (audio matches a known synthetic generator). "
                "Indicates AI-generated audio was played through a physical medium "
                "and re-recorded."
            ),
        )

    # ------------------------------------------------------------------
    # Step 3: CONVENTIONAL_EDITING
    #
    # Required conditions:
    #   F3 = match_found=True           — a known source seed was identified
    #   F3 re-encoding data present     — estimated_reencoding_stages is not None
    #   F1 = REAL                       — explicit clean evidence
    #   F4 = CONSISTENT                 — explicit clean evidence
    #
    # UNKNOWN or INSUFFICIENT_DATA cannot satisfy this rule.
    # ------------------------------------------------------------------
    if (
        _f3_match_found_true(evidence)
        and _f3_has_reencoding_data(evidence)
        and _f1_is_real(evidence)
        and _f4_is_consistent(evidence)
    ):
        stages = evidence.f3.estimated_reencoding_stages
        return _make_result(
            cause=AttributionCause.CONVENTIONAL_EDITING,
            reason=(
                "CONVENTIONAL_EDITING: "
                f"F3=match_found=True (known source seed found, "
                f"matched_seed_id={evidence.f3.matched_seed_id!r}); "
                f"estimated_reencoding_stages={stages} (re-encoding data present); "
                f"F1 verdict={evidence.f1.verdict.value} (explicitly real audio); "
                f"F4 verdict={evidence.f4.verdict.value} (explicitly consistent visuals). "
                "Consistent with conventional post-production editing of a known source."
            ),
        )

    # ------------------------------------------------------------------
    # Step 4: INCONCLUSIVE
    #
    # No specific attribution rule was satisfied.
    # ------------------------------------------------------------------
    active: list[str] = []
    if _f1_has_strong_synthetic(evidence):
        active.append(f"F1={evidence.f1.verdict.value}")
    elif _f1_is_real(evidence):
        active.append("F1=REAL")
    if _f4_is_mismatch(evidence):
        active.append("F4=MISMATCH_DETECTED")
    elif _f4_is_consistent(evidence):
        active.append("F4=CONSISTENT")
    if _f2_is_replayed(evidence):
        active.append("F2=REPLAYED_RECORDING")
    if _f3_match_found_true(evidence):
        active.append("F3=match_found=True")
    if _f3_match_found_false(evidence):
        active.append("F3=match_found=False (absence of source match, not manipulation)")

    signal_summary = (
        f"Active signals: {', '.join(active)}."
        if active
        else "No definitive signals available."
    )

    return _make_result(
        cause=AttributionCause.UNKNOWN,
        reason=(
            f"INCONCLUSIVE: insufficient evidence to reach a specific attribution. "
            f"{signal_summary} "
            f"No attribution rule conditions were fully satisfied."
        ),
    )
