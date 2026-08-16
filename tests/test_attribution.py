"""
Tests for src/p_fusion/attribution.py.

Sources:
    docs/contracts.md      — allowed field names and values
    src/p_fusion/schemas.py — internal types under test
    src/p_fusion/attribution.py — function under test

No invented thresholds, weights, probabilities, conflict rules,
or detector behaviour beyond docs/contracts.md.
"""

import pytest

from p_fusion.attribution import attribute
from p_fusion.schemas import (
    AttributionCause,
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
# Canonical evidence builders
# ---------------------------------------------------------------------------

def _ai_synthesis_evidence() -> F5Evidence:
    """All three required AI_SYNTHESIS signals definitively present."""
    return F5Evidence(
        f1=F1Evidence(
            known_generator_similarity=Level.HIGH,
            natural_speech_consistency=Level.LOW,
            verdict=F1Verdict.UNKNOWN_SYNTHETIC,
            confidence=0.9,
        ),
        f2=F2Evidence(),                 # absent — not required for this rule
        f3=F3Evidence(match_found=False),
        f4=F4Evidence(
            frames_with_usable_eyes=20,
            catchlight_mismatch_flagged_frames=["00:00:04"],
            mismatch_rate=0.5,
            verdict=F4Verdict.MISMATCH_DETECTED,
        ),
    )


def _re_recorded_evidence() -> F5Evidence:
    """Both required signals for AI_GENERATED_THEN_RECORDED definitively present."""
    return F5Evidence(
        f1=F1Evidence(verdict=F1Verdict.KNOWN_SYNTHETIC, confidence=0.8),
        f2=F2Evidence(
            pathway=F2Pathway.REPLAYED_RECORDING,
            reverb_detected=True,
            confidence=0.85,
        ),
        f3=F3Evidence(),
        f4=F4Evidence(),
    )


def _conventional_editing_evidence() -> F5Evidence:
    """All four CONVENTIONAL_EDITING required signals explicit and positive."""
    return F5Evidence(
        f1=F1Evidence(verdict=F1Verdict.REAL, confidence=0.95),
        f2=F2Evidence(),
        f3=F3Evidence(
            match_found=True,
            matched_seed_id="seed_001",
            estimated_reencoding_stages=2,
            metadata=F3Metadata(
                encoder="H.264",
                creation_time="2024-01-01T00:00:00Z",
                resolution="1920x1080",
            ),
        ),
        f4=F4Evidence(
            frames_with_usable_eyes=50,
            catchlight_mismatch_flagged_frames=[],
            mismatch_rate=0.0,
            verdict=F4Verdict.CONSISTENT,
        ),
    )


def _all_unknown_evidence() -> F5Evidence:
    """Default F5Evidence — every field at its UNKNOWN/None default."""
    return F5Evidence()


# ---------------------------------------------------------------------------
# Test 1 — AI_SYNTHESIS
# ---------------------------------------------------------------------------

class TestAISynthesis:
    """Full three-signal combination and each required signal absent individually."""

    def test_full_signal_set_produces_ai_synthesis(self):
        assert attribute(_ai_synthesis_evidence()).cause is AttributionCause.AI_SYNTHESIS

    def test_no_conflict_flag_on_ai_synthesis(self):
        assert attribute(_ai_synthesis_evidence()).conflicts.has_conflict is False

    def test_missing_f4_mismatch_prevents_ai_synthesis(self):
        ev = _ai_synthesis_evidence()
        ev.f4.verdict = F4Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f4_consistent_prevents_ai_synthesis(self):
        ev = _ai_synthesis_evidence()
        ev.f4.verdict = F4Verdict.CONSISTENT
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f3_match_found_none_prevents_ai_synthesis(self):
        # None (absent) is not the same as False (explicit no-match)
        ev = _ai_synthesis_evidence()
        ev.f3.match_found = None
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f3_match_found_true_prevents_ai_synthesis(self):
        ev = _ai_synthesis_evidence()
        ev.f3.match_found = True
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f1_unknown_prevents_ai_synthesis(self):
        ev = _ai_synthesis_evidence()
        ev.f1.verdict = F1Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f1_known_synthetic_prevents_ai_synthesis(self):
        # Rule requires UNKNOWN_SYNTHETIC specifically, not KNOWN_SYNTHETIC
        ev = _ai_synthesis_evidence()
        ev.f1.verdict = F1Verdict.KNOWN_SYNTHETIC
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f1_real_prevents_ai_synthesis(self):
        ev = _ai_synthesis_evidence()
        ev.f1.verdict = F1Verdict.REAL
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS


# ---------------------------------------------------------------------------
# Test 2 — AI_GENERATED_THEN_RECORDED (RE_RECORDED)
# ---------------------------------------------------------------------------

class TestRERecorded:
    """Both required signals and each absent individually."""

    def test_full_signal_set_produces_re_recorded(self):
        assert attribute(_re_recorded_evidence()).cause is AttributionCause.RE_RECORDED

    def test_no_conflict_flag_on_re_recorded(self):
        assert attribute(_re_recorded_evidence()).conflicts.has_conflict is False

    def test_missing_f1_known_synthetic_prevents_re_recorded(self):
        ev = _re_recorded_evidence()
        ev.f1.verdict = F1Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED

    def test_missing_f2_replayed_prevents_re_recorded(self):
        ev = _re_recorded_evidence()
        ev.f2.pathway = F2Pathway.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED

    def test_f1_unknown_synthetic_with_replayed_is_not_re_recorded(self):
        # Rule requires KNOWN_SYNTHETIC; UNKNOWN_SYNTHETIC is a different enum value
        ev = _re_recorded_evidence()
        ev.f1.verdict = F1Verdict.UNKNOWN_SYNTHETIC
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED

    def test_f2_direct_genuine_with_known_synthetic_is_not_re_recorded(self):
        ev = _re_recorded_evidence()
        ev.f2.pathway = F2Pathway.DIRECT_GENUINE
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED

    def test_f2_direct_synthetic_with_known_synthetic_is_not_re_recorded(self):
        ev = _re_recorded_evidence()
        ev.f2.pathway = F2Pathway.DIRECT_SYNTHETIC
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED


# ---------------------------------------------------------------------------
# Test 3 — CONVENTIONAL_EDITING
# ---------------------------------------------------------------------------

class TestConventionalEditing:
    """Full four-signal combination, zero-stage edge case, and each signal absent/wrong."""

    def test_full_signal_set_produces_conventional_editing(self):
        assert attribute(_conventional_editing_evidence()).cause is AttributionCause.CONVENTIONAL_EDITING

    def test_no_conflict_flag_on_conventional_editing(self):
        assert attribute(_conventional_editing_evidence()).conflicts.has_conflict is False

    def test_zero_reencoding_stages_still_conventional_editing(self):
        # 0 stages is valid data (direct copy from a known source)
        ev = _conventional_editing_evidence()
        ev.f3.estimated_reencoding_stages = 0
        assert attribute(ev).cause is AttributionCause.CONVENTIONAL_EDITING

    def test_f1_unknown_prevents_conventional_editing(self):
        # UNKNOWN is neutral — does not satisfy the explicit F1=REAL requirement
        ev = _conventional_editing_evidence()
        ev.f1.verdict = F1Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f4_unknown_prevents_conventional_editing(self):
        # UNKNOWN is neutral — does not satisfy the explicit F4=CONSISTENT requirement
        ev = _conventional_editing_evidence()
        ev.f4.verdict = F4Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f4_mismatch_prevents_conventional_editing(self):
        ev = _conventional_editing_evidence()
        ev.f4.verdict = F4Verdict.MISMATCH_DETECTED
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f3_no_match_prevents_conventional_editing(self):
        ev = _conventional_editing_evidence()
        ev.f3.match_found = False
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f3_match_found_none_prevents_conventional_editing(self):
        ev = _conventional_editing_evidence()
        ev.f3.match_found = None
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_missing_reencoding_stages_prevents_conventional_editing(self):
        ev = _conventional_editing_evidence()
        ev.f3.estimated_reencoding_stages = None
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f1_known_synthetic_prevents_conventional_editing(self):
        ev = _conventional_editing_evidence()
        ev.f1.verdict = F1Verdict.KNOWN_SYNTHETIC
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f1_unknown_synthetic_prevents_conventional_editing(self):
        ev = _conventional_editing_evidence()
        ev.f1.verdict = F1Verdict.UNKNOWN_SYNTHETIC
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING


# ---------------------------------------------------------------------------
# Test 4 — INCONCLUSIVE
# ---------------------------------------------------------------------------

class TestInconclusive:
    """No rule fully satisfied → AttributionCause.UNKNOWN with no conflict."""

    def test_all_unknown_is_inconclusive(self):
        assert attribute(_all_unknown_evidence()).cause is AttributionCause.UNKNOWN

    def test_all_unknown_has_no_conflict(self):
        assert attribute(_all_unknown_evidence()).conflicts.has_conflict is False

    def test_f4_mismatch_alone_is_inconclusive(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED))
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f3_false_and_f4_mismatch_without_f1_is_inconclusive(self):
        ev = F5Evidence(
            f3=F3Evidence(match_found=False),
            f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED),
        )
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f1_unknown_synthetic_and_f4_mismatch_without_f3_is_inconclusive(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC),
            f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED),
        )
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f1_unknown_synthetic_and_f3_false_without_f4_is_inconclusive(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC),
            f3=F3Evidence(match_found=False),
        )
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f2_replayed_alone_is_inconclusive(self):
        ev = F5Evidence(f2=F2Evidence(pathway=F2Pathway.REPLAYED_RECORDING))
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f1_known_synthetic_alone_is_inconclusive(self):
        ev = F5Evidence(f1=F1Evidence(verdict=F1Verdict.KNOWN_SYNTHETIC))
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    @pytest.mark.parametrize("missing", ["f4", "f3", "f1"])
    def test_each_ai_synthesis_signal_missing_is_inconclusive(self, missing):
        """Each two-of-three combination for AI_SYNTHESIS must remain INCONCLUSIVE."""
        ev = _ai_synthesis_evidence()
        if missing == "f4":
            ev.f4.verdict = F4Verdict.UNKNOWN
        elif missing == "f3":
            ev.f3.match_found = None
        else:
            ev.f1.verdict = F1Verdict.UNKNOWN
        result = attribute(ev)
        assert result.cause is AttributionCause.UNKNOWN, (
            f"Expected INCONCLUSIVE when {missing} is absent, got {result.cause}"
        )


# ---------------------------------------------------------------------------
# Test 5 — UNKNOWN evidence never satisfies a positive attribution condition
# ---------------------------------------------------------------------------

class TestUnknownNeutral:
    """UNKNOWN / INSUFFICIENT_DATA values are neutral: they cannot satisfy any rule."""

    def test_f1_unknown_does_not_satisfy_re_recorded_f1(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN),
            f2=F2Evidence(pathway=F2Pathway.REPLAYED_RECORDING),
        )
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED

    def test_f2_unknown_does_not_satisfy_re_recorded_f2(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.KNOWN_SYNTHETIC),
            f2=F2Evidence(pathway=F2Pathway.UNKNOWN),
        )
        assert attribute(ev).cause is not AttributionCause.RE_RECORDED

    def test_f4_unknown_does_not_satisfy_ai_synthesis_f4(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC),
            f3=F3Evidence(match_found=False),
            f4=F4Evidence(verdict=F4Verdict.UNKNOWN),
        )
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f1_unknown_does_not_satisfy_conventional_editing_f1_real(self):
        ev = _conventional_editing_evidence()
        ev.f1.verdict = F1Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f4_unknown_does_not_satisfy_conventional_editing_f4_consistent(self):
        ev = _conventional_editing_evidence()
        ev.f4.verdict = F4Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_f4_insufficient_data_does_not_satisfy_conventional_editing_f4_consistent(self):
        """INSUFFICIENT_DATA is mapped to UNKNOWN by the normalizer; must not satisfy CONSISTENT."""
        ev = _conventional_editing_evidence()
        ev.f4.verdict = F4Verdict.UNKNOWN      # represents post-normalisation INSUFFICIENT_DATA
        assert attribute(ev).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_all_unknown_produces_inconclusive_not_conventional_editing(self):
        assert attribute(_all_unknown_evidence()).cause is not AttributionCause.CONVENTIONAL_EDITING

    def test_all_unknown_produces_inconclusive_not_ai_synthesis(self):
        assert attribute(_all_unknown_evidence()).cause is not AttributionCause.AI_SYNTHESIS

    def test_all_unknown_produces_inconclusive_not_re_recorded(self):
        assert attribute(_all_unknown_evidence()).cause is not AttributionCause.RE_RECORDED


# ---------------------------------------------------------------------------
# Test 6 — F3.match_found=False alone never produces a manipulation attribution
# ---------------------------------------------------------------------------

class TestF3NoMatchNotManipulation:
    """F3 no-match is absence of a known source, not evidence of manipulation."""

    def test_f3_no_match_alone_is_inconclusive(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f3_no_match_alone_has_no_conflict(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        assert attribute(ev).conflicts.has_conflict is False

    def test_f3_no_match_is_not_a_manipulation_cause(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        result = attribute(ev)
        assert result.cause not in (
            AttributionCause.AI_SYNTHESIS,
            AttributionCause.CONVENTIONAL_EDITING,
            AttributionCause.RE_RECORDED,
        )

    def test_f3_no_match_with_f4_mismatch_is_inconclusive(self):
        # Two signals, but F1 is absent — cannot reach AI_SYNTHESIS
        ev = F5Evidence(
            f3=F3Evidence(match_found=False),
            f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED),
        )
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f3_no_match_with_f1_unknown_synthetic_is_inconclusive(self):
        # Two signals, but F4 is absent — cannot reach AI_SYNTHESIS
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC),
            f3=F3Evidence(match_found=False),
        )
        assert attribute(ev).cause is AttributionCause.UNKNOWN

    def test_f3_match_found_none_not_treated_as_explicit_false(self):
        # F3.match_found=None must not count as a no-match signal
        ev_none = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC),
            f3=F3Evidence(match_found=None),
            f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED),
        )
        assert attribute(ev_none).cause is not AttributionCause.AI_SYNTHESIS


# ---------------------------------------------------------------------------
# Test 7 — F4=MISMATCH_DETECTED alone never produces AI_SYNTHESIS
# ---------------------------------------------------------------------------

class TestF4MismatchAlone:
    """F4 is supporting evidence; MISMATCH_DETECTED without the other two signals is INCONCLUSIVE."""

    def test_f4_mismatch_alone_is_not_ai_synthesis(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED))
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f4_mismatch_and_f3_false_without_f1_is_not_ai_synthesis(self):
        ev = F5Evidence(
            f3=F3Evidence(match_found=False),
            f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED),
        )
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    def test_f4_mismatch_and_f1_without_f3_is_not_ai_synthesis(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC),
            f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED),
        )
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS

    @pytest.mark.parametrize("missing", ["f4", "f3", "f1"])
    def test_two_of_three_ai_synthesis_signals_never_sufficient(self, missing):
        ev = _ai_synthesis_evidence()
        if missing == "f4":
            ev.f4.verdict = F4Verdict.UNKNOWN
        elif missing == "f3":
            ev.f3.match_found = None
        else:
            ev.f1.verdict = F1Verdict.UNKNOWN
        assert attribute(ev).cause is not AttributionCause.AI_SYNTHESIS, (
            f"Expected no AI_SYNTHESIS when {missing} is absent"
        )


# ---------------------------------------------------------------------------
# Test 8 — Missing F1/F2/F3/F4 handled safely
# ---------------------------------------------------------------------------

class TestMissingFeaturesSafe:
    """Every combination of a single present feature must not raise."""

    def test_all_defaults_does_not_raise(self):
        result = attribute(F5Evidence())
        assert result is not None

    def test_all_defaults_returns_attribution_cause(self):
        result = attribute(F5Evidence())
        assert isinstance(result.cause, AttributionCause)

    def test_all_defaults_returns_conflict_object(self):
        result = attribute(F5Evidence())
        assert isinstance(result.conflicts.has_conflict, bool)

    def test_only_f1_present_does_not_raise(self):
        ev = F5Evidence(f1=F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC))
        assert attribute(ev) is not None

    def test_only_f2_present_does_not_raise(self):
        ev = F5Evidence(f2=F2Evidence(pathway=F2Pathway.REPLAYED_RECORDING))
        assert attribute(ev) is not None

    def test_only_f3_match_false_does_not_raise(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        assert attribute(ev) is not None

    def test_only_f3_match_true_does_not_raise(self):
        ev = F5Evidence(f3=F3Evidence(match_found=True, estimated_reencoding_stages=1))
        assert attribute(ev) is not None

    def test_only_f4_present_does_not_raise(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.MISMATCH_DETECTED))
        assert attribute(ev) is not None

    def test_f3_with_full_metadata_does_not_raise(self):
        ev = F5Evidence(
            f3=F3Evidence(
                match_found=True,
                matched_seed_id="seed_003",
                estimated_reencoding_stages=3,
                metadata=F3Metadata(
                    encoder="AV1",
                    creation_time="2025-01-01",
                    resolution="3840x2160",
                ),
            )
        )
        assert attribute(ev) is not None


# ---------------------------------------------------------------------------
# Test 9 — Deterministic output for identical evidence
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Identical inputs must always produce the same cause and details."""

    def test_ai_synthesis_same_cause_on_repeat(self):
        ev = _ai_synthesis_evidence()
        assert attribute(ev).cause is attribute(ev).cause

    def test_ai_synthesis_same_details_on_repeat(self):
        ev = _ai_synthesis_evidence()
        assert attribute(ev).conflicts.details == attribute(ev).conflicts.details

    def test_re_recorded_same_cause_on_repeat(self):
        ev = _re_recorded_evidence()
        assert attribute(ev).cause is attribute(ev).cause

    def test_conventional_editing_same_cause_on_repeat(self):
        ev = _conventional_editing_evidence()
        assert attribute(ev).cause is attribute(ev).cause

    def test_all_unknown_always_produces_same_cause(self):
        ev = _all_unknown_evidence()
        assert attribute(ev).cause is attribute(ev).cause

    def test_two_calls_return_independent_result_objects(self):
        ev = _ai_synthesis_evidence()
        r1 = attribute(ev)
        r2 = attribute(ev)
        assert r1 is not r2


# ---------------------------------------------------------------------------
# Test 10 — Reasoning output identifies evidence without inventing measurements
# ---------------------------------------------------------------------------

class TestReasoningOutput:
    """The details string must name the signals that drove the decision.
    It must not contain invented numeric thresholds or probability language."""

    def test_ai_synthesis_details_names_f4_signal(self):
        result = attribute(_ai_synthesis_evidence())
        assert "MISMATCH_DETECTED" in result.conflicts.details

    def test_ai_synthesis_details_names_f3_signal(self):
        result = attribute(_ai_synthesis_evidence())
        assert "match_found=False" in result.conflicts.details

    def test_ai_synthesis_details_names_f1_signal(self):
        result = attribute(_ai_synthesis_evidence())
        assert "UNKNOWN_SYNTHETIC" in result.conflicts.details

    def test_ai_synthesis_details_notes_f3_is_not_standalone_manipulation(self):
        # The reason must acknowledge that F3 no-match is not manipulation evidence alone
        result = attribute(_ai_synthesis_evidence())
        lower = result.conflicts.details.lower()
        assert "not manipulation" in lower or "alone" in lower

    def test_re_recorded_details_names_f2_signal(self):
        result = attribute(_re_recorded_evidence())
        assert "REPLAYED_RECORDING" in result.conflicts.details

    def test_re_recorded_details_names_f1_signal(self):
        result = attribute(_re_recorded_evidence())
        assert "KNOWN_SYNTHETIC" in result.conflicts.details

    def test_conventional_editing_details_names_f3_match(self):
        result = attribute(_conventional_editing_evidence())
        assert "match_found=True" in result.conflicts.details

    def test_conventional_editing_details_names_f1_real(self):
        result = attribute(_conventional_editing_evidence())
        assert "REAL" in result.conflicts.details

    def test_conventional_editing_details_names_f4_consistent(self):
        result = attribute(_conventional_editing_evidence())
        assert "CONSISTENT" in result.conflicts.details

    def test_inconclusive_details_contains_inconclusive_label(self):
        result = attribute(_all_unknown_evidence())
        assert "INCONCLUSIVE" in result.conflicts.details

    def test_details_is_a_non_empty_string_for_all_causes(self):
        for ev in [
            _ai_synthesis_evidence(),
            _re_recorded_evidence(),
            _conventional_editing_evidence(),
            _all_unknown_evidence(),
        ]:
            result = attribute(ev)
            assert isinstance(result.conflicts.details, str)
            assert len(result.conflicts.details) > 0

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_details_contains_no_invented_threshold_language(self, ev):
        details = attribute(ev).conflicts.details.lower()
        for forbidden in ("threshold", "weight", "probability", "score >", "score <"):
            assert forbidden not in details, (
                f"Reason contains invented threshold term {forbidden!r}: {details!r}"
            )
