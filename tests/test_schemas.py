"""
Tests for src/p_fusion/schemas.py.

Validates the internal F5 evidence schema against the contracts in
docs/contracts.md.

NOTE ON TYPE ENFORCEMENT
------------------------
The schemas use plain Python dataclasses, not Pydantic models.  Dataclasses
do NOT enforce type annotations at runtime — assigning a wrong Python type to
a dataclass field silently succeeds.  The only runtime rejection that actually
occurs is via the Enum constructors: `SomeEnum("BAD_VALUE")` raises ValueError.

Tests are written to reflect this reality:
  - Enum-value tests assert that invalid string values raise ValueError.
  - Type-mismatch tests that would silently pass at the dataclass level are
    documented with an explanatory comment rather than a false assertion.
"""

import pytest

from p_fusion.schemas import (
    AttributionCause,
    AttributionConflict,
    F1Evidence,
    F1Verdict,
    F2Evidence,
    F2Pathway,
    F3Evidence,
    F3Metadata,
    F4Evidence,
    F4Verdict,
    F5AttributionResult,
    F5Evidence,
    F5FusionResult,
    FinalVerdict,
    Level,
)


# ===========================================================================
# Level enum  (used by F1 fields)
# ===========================================================================

class TestLevel:
    def test_allowed_values(self):
        assert Level("LOW") is Level.LOW
        assert Level("MED") is Level.MED
        assert Level("HIGH") is Level.HIGH

    def test_unknown_is_member(self):
        assert Level("UNKNOWN") is Level.UNKNOWN

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Level("VERY_HIGH")

    def test_values_are_strings(self):
        # Level inherits str so members compare equal to their string values.
        assert Level.LOW == "LOW"
        assert Level.UNKNOWN == "UNKNOWN"


# ===========================================================================
# F1 — Zero-Day Audio
# ===========================================================================

class TestF1Verdict:
    def test_allowed_values(self):
        assert F1Verdict("REAL") is F1Verdict.REAL
        assert F1Verdict("KNOWN_SYNTHETIC") is F1Verdict.KNOWN_SYNTHETIC
        assert F1Verdict("UNKNOWN_SYNTHETIC") is F1Verdict.UNKNOWN_SYNTHETIC

    def test_unknown_is_member(self):
        assert F1Verdict("UNKNOWN") is F1Verdict.UNKNOWN

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            F1Verdict("FAKE")

    def test_values_match_contract(self):
        # Exact string values from docs/contracts.md
        assert F1Verdict.REAL.value == "REAL"
        assert F1Verdict.KNOWN_SYNTHETIC.value == "KNOWN_SYNTHETIC"
        assert F1Verdict.UNKNOWN_SYNTHETIC.value == "UNKNOWN_SYNTHETIC"


class TestF1Evidence:
    def test_defaults_are_unknown(self):
        ev = F1Evidence()
        assert ev.known_generator_similarity is Level.UNKNOWN
        assert ev.natural_speech_consistency is Level.UNKNOWN
        assert ev.verdict is F1Verdict.UNKNOWN
        assert ev.confidence is None

    def test_valid_full_construction(self):
        ev = F1Evidence(
            known_generator_similarity=Level.HIGH,
            natural_speech_consistency=Level.LOW,
            verdict=F1Verdict.KNOWN_SYNTHETIC,
            confidence=0.91,
        )
        assert ev.known_generator_similarity is Level.HIGH
        assert ev.natural_speech_consistency is Level.LOW
        assert ev.verdict is F1Verdict.KNOWN_SYNTHETIC
        assert ev.confidence == pytest.approx(0.91)

    def test_confidence_boundary_values(self):
        # Contracts state confidence is 0.0–1.0; we can store both extremes.
        assert F1Evidence(confidence=0.0).confidence == pytest.approx(0.0)
        assert F1Evidence(confidence=1.0).confidence == pytest.approx(1.0)

    def test_missing_confidence_is_none(self):
        # Missing data must be representable as UNKNOWN (None here).
        assert F1Evidence().confidence is None

    def test_real_verdict_exists(self):
        ev = F1Evidence(verdict=F1Verdict.REAL)
        assert ev.verdict is F1Verdict.REAL

    def test_unknown_synthetic_verdict(self):
        ev = F1Evidence(verdict=F1Verdict.UNKNOWN_SYNTHETIC)
        assert ev.verdict is F1Verdict.UNKNOWN_SYNTHETIC


# ===========================================================================
# F2 — Replay Detection
# ===========================================================================

class TestF2Pathway:
    def test_allowed_values(self):
        assert F2Pathway("DIRECT_GENUINE") is F2Pathway.DIRECT_GENUINE
        assert F2Pathway("DIRECT_SYNTHETIC") is F2Pathway.DIRECT_SYNTHETIC
        assert F2Pathway("REPLAYED_RECORDING") is F2Pathway.REPLAYED_RECORDING

    def test_unknown_is_member(self):
        assert F2Pathway("UNKNOWN") is F2Pathway.UNKNOWN

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            F2Pathway("RE_RECORDED")

    def test_values_match_contract(self):
        assert F2Pathway.DIRECT_GENUINE.value == "DIRECT_GENUINE"
        assert F2Pathway.DIRECT_SYNTHETIC.value == "DIRECT_SYNTHETIC"
        assert F2Pathway.REPLAYED_RECORDING.value == "REPLAYED_RECORDING"


class TestF2Evidence:
    def test_defaults_are_unknown(self):
        ev = F2Evidence()
        assert ev.pathway is F2Pathway.UNKNOWN
        assert ev.reverb_detected is None
        assert ev.confidence is None

    def test_valid_full_construction(self):
        ev = F2Evidence(
            pathway=F2Pathway.REPLAYED_RECORDING,
            reverb_detected=True,
            confidence=0.78,
        )
        assert ev.pathway is F2Pathway.REPLAYED_RECORDING
        assert ev.reverb_detected is True
        assert ev.confidence == pytest.approx(0.78)

    def test_reverb_false(self):
        ev = F2Evidence(reverb_detected=False)
        assert ev.reverb_detected is False

    def test_missing_data_representable(self):
        ev = F2Evidence()
        assert ev.reverb_detected is None
        assert ev.confidence is None


# ===========================================================================
# F3 — Source / Propagation
# ===========================================================================

class TestF3Metadata:
    def test_all_fields_optional(self):
        meta = F3Metadata()
        assert meta.encoder is None
        assert meta.creation_time is None
        assert meta.resolution is None

    def test_valid_construction(self):
        meta = F3Metadata(
            encoder="libx264",
            creation_time="2024-01-01T00:00:00",
            resolution="1920x1080",
        )
        assert meta.encoder == "libx264"
        assert meta.creation_time == "2024-01-01T00:00:00"
        assert meta.resolution == "1920x1080"


class TestF3Evidence:
    def test_defaults_are_none(self):
        ev = F3Evidence()
        assert ev.match_found is None
        assert ev.matched_seed_id is None
        assert ev.estimated_reencoding_stages is None
        assert ev.metadata is None

    def test_match_found_true(self):
        ev = F3Evidence(match_found=True, matched_seed_id="seed_003")
        assert ev.match_found is True
        assert ev.matched_seed_id == "seed_003"

    def test_match_found_false_no_seed_id(self):
        """
        Contract: match_found=False means NO MATCH — not proof of manipulation.
        The seed ID must be None when there is no match.
        """
        ev = F3Evidence(match_found=False, matched_seed_id=None)
        assert ev.match_found is False
        assert ev.matched_seed_id is None

    def test_no_match_does_not_imply_manipulation(self):
        """
        Semantic assertion: the schema must be able to represent a no-match
        state independently of any manipulation verdict.  F3 alone carries
        no manipulation verdict field — that decision belongs to F5.
        """
        ev = F3Evidence(match_found=False)
        # F3Evidence has no 'manipulation' or 'verdict' field.
        assert not hasattr(ev, "verdict")
        assert not hasattr(ev, "manipulation")

    def test_reencoding_stages_is_integer(self):
        ev = F3Evidence(estimated_reencoding_stages=3)
        assert ev.estimated_reencoding_stages == 3

    def test_reencoding_stages_zero(self):
        ev = F3Evidence(estimated_reencoding_stages=0)
        assert ev.estimated_reencoding_stages == 0

    def test_with_metadata(self):
        meta = F3Metadata(encoder="h264", resolution="720p")
        ev = F3Evidence(match_found=True, matched_seed_id="seed_001", metadata=meta)
        assert ev.metadata.encoder == "h264"
        assert ev.metadata.resolution == "720p"

    def test_missing_data_representable(self):
        ev = F3Evidence()
        # All fields must be None when no data is available.
        assert ev.match_found is None
        assert ev.matched_seed_id is None
        assert ev.estimated_reencoding_stages is None
        assert ev.metadata is None


# ===========================================================================
# F4 — Eye Reflection
# ===========================================================================

class TestF4Verdict:
    def test_allowed_values(self):
        assert F4Verdict("CONSISTENT") is F4Verdict.CONSISTENT
        assert F4Verdict("MISMATCH_DETECTED") is F4Verdict.MISMATCH_DETECTED
        assert F4Verdict("INSUFFICIENT_DATA") is F4Verdict.INSUFFICIENT_DATA

    def test_unknown_is_member(self):
        assert F4Verdict("UNKNOWN") is F4Verdict.UNKNOWN

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            F4Verdict("MANIPULATED")

    def test_insufficient_data_distinct_from_consistent(self):
        """
        Contract: INSUFFICIENT_DATA must remain distinguishable from CONSISTENT.
        They must be separate enum members that do not compare equal.
        """
        assert F4Verdict.INSUFFICIENT_DATA is not F4Verdict.CONSISTENT
        assert F4Verdict.INSUFFICIENT_DATA != F4Verdict.CONSISTENT

    def test_values_match_contract(self):
        assert F4Verdict.CONSISTENT.value == "CONSISTENT"
        assert F4Verdict.MISMATCH_DETECTED.value == "MISMATCH_DETECTED"
        assert F4Verdict.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"


class TestF4Evidence:
    def test_defaults(self):
        ev = F4Evidence()
        assert ev.frames_with_usable_eyes is None
        assert ev.catchlight_mismatch_flagged_frames == []
        assert ev.mismatch_rate is None
        assert ev.verdict is F4Verdict.UNKNOWN

    def test_valid_full_construction(self):
        ev = F4Evidence(
            frames_with_usable_eyes=42,
            catchlight_mismatch_flagged_frames=["00:00:04", "00:00:08"],
            mismatch_rate=0.15,
            verdict=F4Verdict.MISMATCH_DETECTED,
        )
        assert ev.frames_with_usable_eyes == 42
        assert ev.catchlight_mismatch_flagged_frames == ["00:00:04", "00:00:08"]
        assert ev.mismatch_rate == pytest.approx(0.15)
        assert ev.verdict is F4Verdict.MISMATCH_DETECTED

    def test_consistent_verdict(self):
        ev = F4Evidence(verdict=F4Verdict.CONSISTENT, mismatch_rate=0.0)
        assert ev.verdict is F4Verdict.CONSISTENT

    def test_insufficient_data_verdict(self):
        """
        INSUFFICIENT_DATA must be stored and retrieved without coercion to
        CONSISTENT or UNKNOWN.
        """
        ev = F4Evidence(
            frames_with_usable_eyes=0,
            verdict=F4Verdict.INSUFFICIENT_DATA,
        )
        assert ev.verdict is F4Verdict.INSUFFICIENT_DATA
        # Must not equal CONSISTENT
        assert ev.verdict != F4Verdict.CONSISTENT
        # Must not equal UNKNOWN
        assert ev.verdict != F4Verdict.UNKNOWN

    def test_no_flagged_frames_by_default(self):
        ev = F4Evidence()
        # Default must be an empty list, not None, matching contract array type.
        assert isinstance(ev.catchlight_mismatch_flagged_frames, list)
        assert len(ev.catchlight_mismatch_flagged_frames) == 0

    def test_flagged_frames_are_independent_per_instance(self):
        # Guard against mutable default argument anti-pattern.
        ev1 = F4Evidence()
        ev2 = F4Evidence()
        ev1.catchlight_mismatch_flagged_frames.append("00:00:01")
        assert ev2.catchlight_mismatch_flagged_frames == []

    def test_mismatch_rate_zero(self):
        ev = F4Evidence(mismatch_rate=0.0)
        assert ev.mismatch_rate == pytest.approx(0.0)

    def test_mismatch_rate_one(self):
        ev = F4Evidence(mismatch_rate=1.0)
        assert ev.mismatch_rate == pytest.approx(1.0)

    def test_missing_data_representable(self):
        ev = F4Evidence()
        assert ev.frames_with_usable_eyes is None
        assert ev.mismatch_rate is None


# ===========================================================================
# F5 combined evidence container
# ===========================================================================

class TestF5Evidence:
    def test_defaults_produce_all_unknown_sub_evidence(self):
        ev = F5Evidence()
        assert ev.f1.verdict is F1Verdict.UNKNOWN
        assert ev.f2.pathway is F2Pathway.UNKNOWN
        assert ev.f3.match_found is None
        assert ev.f4.verdict is F4Verdict.UNKNOWN

    def test_sub_evidence_objects_are_independent(self):
        """Each F5Evidence instance must have its own sub-evidence objects."""
        ev1 = F5Evidence()
        ev2 = F5Evidence()
        ev1.f1.confidence = 0.5
        assert ev2.f1.confidence is None

    def test_accepts_populated_sub_evidence(self):
        ev = F5Evidence(
            f1=F1Evidence(verdict=F1Verdict.REAL, confidence=0.99),
            f2=F2Evidence(pathway=F2Pathway.DIRECT_GENUINE),
            f3=F3Evidence(match_found=False),
            f4=F4Evidence(verdict=F4Verdict.INSUFFICIENT_DATA),
        )
        assert ev.f1.verdict is F1Verdict.REAL
        assert ev.f2.pathway is F2Pathway.DIRECT_GENUINE
        assert ev.f3.match_found is False
        assert ev.f4.verdict is F4Verdict.INSUFFICIENT_DATA


# ===========================================================================
# Attribution and final fusion result
# ===========================================================================

class TestAttributionCause:
    def test_allowed_values(self):
        assert AttributionCause("AI_SYNTHESIS") is AttributionCause.AI_SYNTHESIS
        assert AttributionCause("CONVENTIONAL_EDITING") is AttributionCause.CONVENTIONAL_EDITING
        assert AttributionCause("RE_RECORDED") is AttributionCause.RE_RECORDED

    def test_unknown_is_member(self):
        assert AttributionCause("UNKNOWN") is AttributionCause.UNKNOWN

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AttributionCause("DEEPFAKE")


class TestFinalVerdict:
    def test_allowed_values(self):
        assert FinalVerdict("REAL") is FinalVerdict.REAL
        assert FinalVerdict("MANIPULATED") is FinalVerdict.MANIPULATED

    def test_unknown_is_member(self):
        assert FinalVerdict("UNKNOWN") is FinalVerdict.UNKNOWN

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            FinalVerdict("SYNTHETIC")


class TestF5FusionResult:
    def test_defaults(self):
        result = F5FusionResult()
        assert result.final_verdict is FinalVerdict.UNKNOWN
        assert result.attribution.cause is AttributionCause.UNKNOWN
        assert result.attribution.conflicts.has_conflict is False
        assert result.evidence.f1.verdict is F1Verdict.UNKNOWN

    def test_conflict_flag(self):
        conflict = AttributionConflict(has_conflict=True, details="F1 and F2 disagree")
        attr = F5AttributionResult(cause=AttributionCause.UNKNOWN, conflicts=conflict)
        result = F5FusionResult(final_verdict=FinalVerdict.UNKNOWN, attribution=attr)
        assert result.attribution.conflicts.has_conflict is True
        assert result.attribution.conflicts.details == "F1 and F2 disagree"

    def test_no_conflict_by_default(self):
        result = F5FusionResult()
        assert result.attribution.conflicts.has_conflict is False
        assert result.attribution.conflicts.details == ""


# ===========================================================================
# Enum rejection — invalid string values
# ===========================================================================

class TestEnumRejection:
    """All Enum constructors must raise ValueError for unlisted string values."""

    @pytest.mark.parametrize("bad", ["VERY_HIGH", "NONE", "", "low", "unknown"])
    def test_level_rejects_bad_value(self, bad):
        with pytest.raises(ValueError):
            Level(bad)

    @pytest.mark.parametrize("bad", ["SYNTHETIC", "FAKE", "AI", ""])
    def test_f1_verdict_rejects_bad_value(self, bad):
        with pytest.raises(ValueError):
            F1Verdict(bad)

    @pytest.mark.parametrize("bad", ["GENUINE", "REPLAY", "UNKNOWN_PATHWAY", ""])
    def test_f2_pathway_rejects_bad_value(self, bad):
        with pytest.raises(ValueError):
            F2Pathway(bad)

    @pytest.mark.parametrize("bad", ["MANIPULATED", "FAKE", "MISMATCH", ""])
    def test_f4_verdict_rejects_bad_value(self, bad):
        with pytest.raises(ValueError):
            F4Verdict(bad)

    @pytest.mark.parametrize("bad", ["REAL", "FAKE", "DEEPFAKE", ""])
    def test_attribution_cause_rejects_bad_value(self, bad):
        with pytest.raises(ValueError):
            AttributionCause(bad)

    @pytest.mark.parametrize("bad", ["CONFIRMED", "SUSPICIOUS", ""])
    def test_final_verdict_rejects_bad_value(self, bad):
        with pytest.raises(ValueError):
            FinalVerdict(bad)
