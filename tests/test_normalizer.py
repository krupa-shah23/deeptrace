"""
Tests for src/p_fusion/evidence_normalizer.py.

Contract source: docs/contracts.md
Schema source:   src/p_fusion/schemas.py

These tests cover the full normalization pipeline:
    raw F1/F2/F3/F4 JSON dicts → F5Evidence

No attribution logic is tested here.
No fields are invented beyond those in docs/contracts.md.
"""

import pytest

from p_fusion.evidence_normalizer import normalize_evidence
from p_fusion.schemas import (
    F1Verdict,
    F2Pathway,
    F4Verdict,
    Level,
)


# ===========================================================================
# Canonical fixture dicts — exact field names from docs/contracts.md
# ===========================================================================

FULL_F1 = {
    "known_generator_similarity": "HIGH",
    "natural_speech_consistency": "LOW",
    "verdict": "KNOWN_SYNTHETIC",
    "confidence": 0.87,
}

FULL_F2 = {
    "pathway": "REPLAYED_RECORDING",
    "reverb_detected": True,
    "confidence": 0.72,
}

FULL_F3 = {
    "match_found": True,
    "matched_seed_id": "seed_003",
    "estimated_reencoding_stages": 2,
    "metadata": {
        "encoder": "libx264",
        "creation_time": "2024-01-15T10:30:00",
        "resolution": "1920x1080",
    },
}

FULL_F4 = {
    "frames_with_usable_eyes": 42,
    "catchlight_mismatch_flagged_frames": ["00:00:04", "00:00:08"],
    "mismatch_rate": 0.15,
    "verdict": "MISMATCH_DETECTED",
}


# ===========================================================================
# Helper
# ===========================================================================

def norm(**kwargs):
    """Thin wrapper so tests can pass only the args they care about."""
    return normalize_evidence(
        f1_data=kwargs.get("f1"),
        f2_data=kwargs.get("f2"),
        f3_data=kwargs.get("f3"),
        f4_data=kwargs.get("f4"),
    )


# ===========================================================================
# All F1-F4 features present
# ===========================================================================

class TestFullEvidence:
    def test_returns_f5evidence(self):
        ev = norm(f1=FULL_F1, f2=FULL_F2, f3=FULL_F3, f4=FULL_F4)
        from p_fusion.schemas import F5Evidence
        assert isinstance(ev, F5Evidence)

    def test_f1_all_fields_preserved(self):
        ev = norm(f1=FULL_F1)
        assert ev.f1.known_generator_similarity is Level.HIGH
        assert ev.f1.natural_speech_consistency is Level.LOW
        assert ev.f1.verdict is F1Verdict.KNOWN_SYNTHETIC
        assert ev.f1.confidence == pytest.approx(0.87)

    def test_f2_all_fields_preserved(self):
        ev = norm(f2=FULL_F2)
        assert ev.f2.pathway is F2Pathway.REPLAYED_RECORDING
        assert ev.f2.reverb_detected is True
        assert ev.f2.confidence == pytest.approx(0.72)

    def test_f3_all_fields_preserved(self):
        ev = norm(f3=FULL_F3)
        assert ev.f3.match_found is True
        assert ev.f3.matched_seed_id == "seed_003"
        assert ev.f3.estimated_reencoding_stages == 2
        assert ev.f3.metadata is not None
        assert ev.f3.metadata.encoder == "libx264"
        assert ev.f3.metadata.creation_time == "2024-01-15T10:30:00"
        assert ev.f3.metadata.resolution == "1920x1080"

    def test_f4_all_fields_preserved(self):
        ev = norm(f4=FULL_F4)
        assert ev.f4.frames_with_usable_eyes == 42
        assert ev.f4.catchlight_mismatch_flagged_frames == ["00:00:04", "00:00:08"]
        assert ev.f4.mismatch_rate == pytest.approx(0.15)
        assert ev.f4.verdict is F4Verdict.MISMATCH_DETECTED

    def test_all_detectors_together(self):
        ev = norm(f1=FULL_F1, f2=FULL_F2, f3=FULL_F3, f4=FULL_F4)
        assert ev.f1.verdict is F1Verdict.KNOWN_SYNTHETIC
        assert ev.f2.pathway is F2Pathway.REPLAYED_RECORDING
        assert ev.f3.match_found is True
        assert ev.f4.verdict is F4Verdict.MISMATCH_DETECTED


# ===========================================================================
# Each detector missing independently → UNKNOWN / None
# ===========================================================================

class TestMissingDetectors:
    def test_f1_missing_all_unknown(self):
        ev = norm()
        assert ev.f1.known_generator_similarity is Level.UNKNOWN
        assert ev.f1.natural_speech_consistency is Level.UNKNOWN
        assert ev.f1.verdict is F1Verdict.UNKNOWN
        assert ev.f1.confidence is None

    def test_f2_missing_all_unknown(self):
        ev = norm()
        assert ev.f2.pathway is F2Pathway.UNKNOWN
        assert ev.f2.reverb_detected is None
        assert ev.f2.confidence is None

    def test_f3_missing_all_none(self):
        ev = norm()
        assert ev.f3.match_found is None
        assert ev.f3.matched_seed_id is None
        assert ev.f3.estimated_reencoding_stages is None
        assert ev.f3.metadata is None

    def test_f4_missing_all_unknown(self):
        ev = norm()
        assert ev.f4.verdict is F4Verdict.UNKNOWN
        assert ev.f4.frames_with_usable_eyes is None
        assert ev.f4.mismatch_rate is None
        assert ev.f4.catchlight_mismatch_flagged_frames == []

    def test_f1_only_others_unknown(self):
        ev = norm(f1=FULL_F1)
        assert ev.f2.pathway is F2Pathway.UNKNOWN
        assert ev.f3.match_found is None
        assert ev.f4.verdict is F4Verdict.UNKNOWN

    def test_f4_only_others_unknown(self):
        ev = norm(f4=FULL_F4)
        assert ev.f1.verdict is F1Verdict.UNKNOWN
        assert ev.f2.pathway is F2Pathway.UNKNOWN
        assert ev.f3.match_found is None


# ===========================================================================
# F1 individual field missing
# ===========================================================================

class TestF1FieldsMissing:
    def test_missing_known_generator_similarity(self):
        data = {k: v for k, v in FULL_F1.items() if k != "known_generator_similarity"}
        ev = norm(f1=data)
        assert ev.f1.known_generator_similarity is Level.UNKNOWN
        # Other fields still populated
        assert ev.f1.verdict is F1Verdict.KNOWN_SYNTHETIC

    def test_missing_natural_speech_consistency(self):
        data = {k: v for k, v in FULL_F1.items() if k != "natural_speech_consistency"}
        ev = norm(f1=data)
        assert ev.f1.natural_speech_consistency is Level.UNKNOWN
        assert ev.f1.known_generator_similarity is Level.HIGH

    def test_missing_verdict(self):
        data = {k: v for k, v in FULL_F1.items() if k != "verdict"}
        ev = norm(f1=data)
        assert ev.f1.verdict is F1Verdict.UNKNOWN

    def test_missing_confidence(self):
        data = {k: v for k, v in FULL_F1.items() if k != "confidence"}
        ev = norm(f1=data)
        assert ev.f1.confidence is None


# ===========================================================================
# F2 individual field missing
# ===========================================================================

class TestF2FieldsMissing:
    def test_missing_pathway(self):
        data = {k: v for k, v in FULL_F2.items() if k != "pathway"}
        ev = norm(f2=data)
        assert ev.f2.pathway is F2Pathway.UNKNOWN

    def test_missing_reverb_detected(self):
        data = {k: v for k, v in FULL_F2.items() if k != "reverb_detected"}
        ev = norm(f2=data)
        assert ev.f2.reverb_detected is None

    def test_missing_confidence(self):
        data = {k: v for k, v in FULL_F2.items() if k != "confidence"}
        ev = norm(f2=data)
        assert ev.f2.confidence is None


# ===========================================================================
# F3 individual field missing
# ===========================================================================

class TestF3FieldsMissing:
    def test_missing_match_found(self):
        data = {k: v for k, v in FULL_F3.items() if k != "match_found"}
        ev = norm(f3=data)
        assert ev.f3.match_found is None

    def test_missing_matched_seed_id(self):
        data = {k: v for k, v in FULL_F3.items() if k != "matched_seed_id"}
        ev = norm(f3=data)
        assert ev.f3.matched_seed_id is None

    def test_missing_estimated_reencoding_stages(self):
        data = {k: v for k, v in FULL_F3.items() if k != "estimated_reencoding_stages"}
        ev = norm(f3=data)
        assert ev.f3.estimated_reencoding_stages is None

    def test_missing_metadata(self):
        data = {k: v for k, v in FULL_F3.items() if k != "metadata"}
        ev = norm(f3=data)
        assert ev.f3.metadata is None


# ===========================================================================
# F4 individual field missing
# ===========================================================================

class TestF4FieldsMissing:
    def test_missing_frames_with_usable_eyes(self):
        data = {k: v for k, v in FULL_F4.items() if k != "frames_with_usable_eyes"}
        ev = norm(f4=data)
        assert ev.f4.frames_with_usable_eyes is None

    def test_missing_catchlight_flagged_frames(self):
        data = {k: v for k, v in FULL_F4.items()
                if k != "catchlight_mismatch_flagged_frames"}
        ev = norm(f4=data)
        assert ev.f4.catchlight_mismatch_flagged_frames == []

    def test_missing_mismatch_rate(self):
        data = {k: v for k, v in FULL_F4.items() if k != "mismatch_rate"}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate is None

    def test_missing_verdict(self):
        data = {k: v for k, v in FULL_F4.items() if k != "verdict"}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.UNKNOWN


# ===========================================================================
# F4 INSUFFICIENT_DATA semantics
# ===========================================================================

class TestF4InsufficientData:
    def test_insufficient_data_verdict_becomes_unknown(self):
        """
        Contract: INSUFFICIENT_DATA → normalized verdict is UNKNOWN.
        The detector could not gather enough frames; no positive evidence.
        """
        data = {**FULL_F4, "verdict": "INSUFFICIENT_DATA"}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.UNKNOWN

    def test_insufficient_data_not_consistent(self):
        data = {**FULL_F4, "verdict": "INSUFFICIENT_DATA"}
        ev = norm(f4=data)
        assert ev.f4.verdict is not F4Verdict.CONSISTENT

    def test_insufficient_data_mismatch_rate_preserved(self):
        """
        Even when verdict is INSUFFICIENT_DATA, mismatch_rate is preserved
        verbatim for reporting purposes.
        """
        data = {**FULL_F4, "verdict": "INSUFFICIENT_DATA", "mismatch_rate": 0.05}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate == pytest.approx(0.05)

    def test_insufficient_data_frames_preserved(self):
        """
        frames_with_usable_eyes is preserved even when verdict is INSUFFICIENT_DATA.
        """
        data = {**FULL_F4, "verdict": "INSUFFICIENT_DATA", "frames_with_usable_eyes": 3}
        ev = norm(f4=data)
        assert ev.f4.frames_with_usable_eyes == 3

    def test_consistent_verdict_passes_through(self):
        data = {**FULL_F4, "verdict": "CONSISTENT"}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.CONSISTENT

    def test_mismatch_detected_passes_through(self):
        ev = norm(f4=FULL_F4)
        assert ev.f4.verdict is F4Verdict.MISMATCH_DETECTED


# ===========================================================================
# F3 match_found=False semantics
# ===========================================================================

class TestF3NoMatch:
    def test_match_found_false_stored_as_false(self):
        """
        Contract: match_found=False means NO MATCH, not proof of manipulation.
        The normalizer must store it as False without adding any manipulation flag.
        """
        data = {**FULL_F3, "match_found": False, "matched_seed_id": None}
        ev = norm(f3=data)
        assert ev.f3.match_found is False

    def test_match_found_false_seed_id_is_none(self):
        data = {**FULL_F3, "match_found": False, "matched_seed_id": None}
        ev = norm(f3=data)
        assert ev.f3.matched_seed_id is None

    def test_f3_evidence_has_no_manipulation_verdict_field(self):
        """
        F3Evidence has no verdict or manipulation field — confirmed by schemas.py.
        A no-match cannot be stored as manipulation evidence.
        """
        data = {**FULL_F3, "match_found": False}
        ev = norm(f3=data)
        assert not hasattr(ev.f3, "verdict")
        assert not hasattr(ev.f3, "manipulation")

    def test_match_found_true(self):
        ev = norm(f3=FULL_F3)
        assert ev.f3.match_found is True
        assert ev.f3.matched_seed_id == "seed_003"


# ===========================================================================
# F4 numeric field preservation
# ===========================================================================

class TestF4NumericPreservation:
    def test_mismatch_rate_boundary_zero(self):
        data = {**FULL_F4, "mismatch_rate": 0.0}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate == pytest.approx(0.0)

    def test_mismatch_rate_boundary_one(self):
        data = {**FULL_F4, "mismatch_rate": 1.0}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate == pytest.approx(1.0)

    def test_frames_with_usable_eyes_zero(self):
        data = {**FULL_F4, "frames_with_usable_eyes": 0}
        ev = norm(f4=data)
        assert ev.f4.frames_with_usable_eyes == 0

    def test_frames_with_usable_eyes_100(self):
        """Upper bound from contracts.md: 0-100 frames."""
        data = {**FULL_F4, "frames_with_usable_eyes": 100}
        ev = norm(f4=data)
        assert ev.f4.frames_with_usable_eyes == 100

    def test_catchlight_timestamp_list_preserved(self):
        timestamps = ["00:00:04", "00:00:08", "00:00:12"]
        data = {**FULL_F4, "catchlight_mismatch_flagged_frames": timestamps}
        ev = norm(f4=data)
        assert ev.f4.catchlight_mismatch_flagged_frames == timestamps

    def test_empty_catchlight_list(self):
        data = {**FULL_F4, "catchlight_mismatch_flagged_frames": []}
        ev = norm(f4=data)
        assert ev.f4.catchlight_mismatch_flagged_frames == []


# ===========================================================================
# F3 metadata preservation
# ===========================================================================

class TestF3MetadataPreservation:
    def test_all_metadata_fields_preserved(self):
        ev = norm(f3=FULL_F3)
        assert ev.f3.metadata.encoder == "libx264"
        assert ev.f3.metadata.creation_time == "2024-01-15T10:30:00"
        assert ev.f3.metadata.resolution == "1920x1080"

    def test_partial_metadata_preserved(self):
        data = {
            **FULL_F3,
            "metadata": {"encoder": "h265", "creation_time": None, "resolution": None},
        }
        ev = norm(f3=data)
        assert ev.f3.metadata.encoder == "h265"
        assert ev.f3.metadata.creation_time is None
        assert ev.f3.metadata.resolution is None

    def test_metadata_none_values_are_none(self):
        data = {
            **FULL_F3,
            "metadata": {"encoder": None, "creation_time": None, "resolution": None},
        }
        ev = norm(f3=data)
        assert ev.f3.metadata.encoder is None
        assert ev.f3.metadata.creation_time is None
        assert ev.f3.metadata.resolution is None

    def test_estimated_reencoding_stages_preserved(self):
        data = {**FULL_F3, "estimated_reencoding_stages": 5}
        ev = norm(f3=data)
        assert ev.f3.estimated_reencoding_stages == 5

    def test_estimated_reencoding_stages_zero(self):
        data = {**FULL_F3, "estimated_reencoding_stages": 0}
        ev = norm(f3=data)
        assert ev.f3.estimated_reencoding_stages == 0


# ===========================================================================
# Out-of-range numeric values → rejected (None)
# ===========================================================================

class TestOutOfRangeValues:
    def test_confidence_above_1_rejected(self):
        data = {**FULL_F1, "confidence": 1.1}
        ev = norm(f1=data)
        assert ev.f1.confidence is None

    def test_confidence_below_0_rejected(self):
        data = {**FULL_F1, "confidence": -0.1}
        ev = norm(f1=data)
        assert ev.f1.confidence is None

    def test_f2_confidence_above_1_rejected(self):
        data = {**FULL_F2, "confidence": 2.0}
        ev = norm(f2=data)
        assert ev.f2.confidence is None

    def test_mismatch_rate_above_1_rejected(self):
        data = {**FULL_F4, "mismatch_rate": 1.5}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate is None

    def test_mismatch_rate_negative_rejected(self):
        data = {**FULL_F4, "mismatch_rate": -0.5}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate is None

    def test_frames_with_usable_eyes_negative_rejected(self):
        """Contracts state 0-100 frames; negative values are rejected."""
        data = {**FULL_F4, "frames_with_usable_eyes": -1}
        ev = norm(f4=data)
        assert ev.f4.frames_with_usable_eyes is None

    def test_reencoding_stages_negative_rejected(self):
        """Contracts state 0-N stages; negative values are rejected."""
        data = {**FULL_F3, "estimated_reencoding_stages": -3}
        ev = norm(f3=data)
        assert ev.f3.estimated_reencoding_stages is None


# ===========================================================================
# Wrong field types do not crash the normalizer
# ===========================================================================

class TestWrongFieldTypes:
    def test_f1_confidence_is_string(self):
        data = {**FULL_F1, "confidence": "high"}
        ev = norm(f1=data)
        assert ev.f1.confidence is None

    def test_f1_verdict_is_integer(self):
        data = {**FULL_F1, "verdict": 1}
        ev = norm(f1=data)
        assert ev.f1.verdict is F1Verdict.UNKNOWN

    def test_f1_level_is_integer(self):
        data = {**FULL_F1, "known_generator_similarity": 99}
        ev = norm(f1=data)
        assert ev.f1.known_generator_similarity is Level.UNKNOWN

    def test_f2_pathway_is_boolean(self):
        data = {**FULL_F2, "pathway": True}
        ev = norm(f2=data)
        assert ev.f2.pathway is F2Pathway.UNKNOWN

    def test_f2_reverb_detected_is_string(self):
        data = {**FULL_F2, "reverb_detected": "yes"}
        ev = norm(f2=data)
        assert ev.f2.reverb_detected is None

    def test_f3_match_found_is_string(self):
        data = {**FULL_F3, "match_found": "true"}
        ev = norm(f3=data)
        assert ev.f3.match_found is None

    def test_f3_estimated_stages_is_float(self):
        """Floats are not valid for an integer field; should be rejected."""
        data = {**FULL_F3, "estimated_reencoding_stages": 2.5}
        ev = norm(f3=data)
        assert ev.f3.estimated_reencoding_stages is None

    def test_f3_estimated_stages_is_boolean(self):
        """True/False must not be accepted as 1/0 for integer fields."""
        data = {**FULL_F3, "estimated_reencoding_stages": True}
        ev = norm(f3=data)
        assert ev.f3.estimated_reencoding_stages is None

    def test_f4_verdict_is_integer(self):
        data = {**FULL_F4, "verdict": 0}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.UNKNOWN

    def test_f4_frames_is_boolean(self):
        """True must not be silently converted to 1."""
        data = {**FULL_F4, "frames_with_usable_eyes": True}
        ev = norm(f4=data)
        assert ev.f4.frames_with_usable_eyes is None

    def test_f4_mismatch_rate_is_string(self):
        data = {**FULL_F4, "mismatch_rate": "0.15"}
        ev = norm(f4=data)
        assert ev.f4.mismatch_rate is None

    def test_f4_catchlight_list_drops_non_strings(self):
        data = {**FULL_F4, "catchlight_mismatch_flagged_frames": ["00:00:04", 999, None, "00:00:08"]}
        ev = norm(f4=data)
        assert ev.f4.catchlight_mismatch_flagged_frames == ["00:00:04", "00:00:08"]

    def test_f4_catchlight_not_a_list(self):
        data = {**FULL_F4, "catchlight_mismatch_flagged_frames": "00:00:04"}
        ev = norm(f4=data)
        assert ev.f4.catchlight_mismatch_flagged_frames == []

    def test_f3_metadata_not_a_dict(self):
        data = {**FULL_F3, "metadata": "invalid"}
        ev = norm(f3=data)
        assert ev.f3.metadata is None

    def test_f1_confidence_is_boolean(self):
        """True/False must not be accepted as 1.0/0.0 for float fields."""
        data = {**FULL_F1, "confidence": True}
        ev = norm(f1=data)
        assert ev.f1.confidence is None

    def test_entirely_wrong_type_does_not_crash(self):
        """An entirely empty dict must not raise; all fields fall back to UNKNOWN."""
        ev = norm(f1={}, f2={}, f3={}, f4={})
        assert ev.f1.verdict is F1Verdict.UNKNOWN
        assert ev.f2.pathway is F2Pathway.UNKNOWN
        assert ev.f3.match_found is None
        assert ev.f4.verdict is F4Verdict.UNKNOWN

    def test_unrecognised_f1_verdict_string(self):
        data = {**FULL_F1, "verdict": "MAYBE_SYNTHETIC"}
        ev = norm(f1=data)
        assert ev.f1.verdict is F1Verdict.UNKNOWN

    def test_unrecognised_f2_pathway_string(self):
        data = {**FULL_F2, "pathway": "WIFI_STREAM"}
        ev = norm(f2=data)
        assert ev.f2.pathway is F2Pathway.UNKNOWN

    def test_unrecognised_f4_verdict_string(self):
        data = {**FULL_F4, "verdict": "MAYBE"}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.UNKNOWN


# ===========================================================================
# Normalization is deterministic
# ===========================================================================

class TestDeterminism:
    def test_same_input_same_output_f1(self):
        ev1 = norm(f1=FULL_F1)
        ev2 = norm(f1=FULL_F1)
        assert ev1.f1.verdict == ev2.f1.verdict
        assert ev1.f1.confidence == ev2.f1.confidence
        assert ev1.f1.known_generator_similarity == ev2.f1.known_generator_similarity
        assert ev1.f1.natural_speech_consistency == ev2.f1.natural_speech_consistency

    def test_same_input_same_output_f4(self):
        ev1 = norm(f4=FULL_F4)
        ev2 = norm(f4=FULL_F4)
        assert ev1.f4.verdict == ev2.f4.verdict
        assert ev1.f4.mismatch_rate == ev2.f4.mismatch_rate
        assert ev1.f4.frames_with_usable_eyes == ev2.f4.frames_with_usable_eyes
        assert ev1.f4.catchlight_mismatch_flagged_frames == ev2.f4.catchlight_mismatch_flagged_frames

    def test_instances_are_independent(self):
        """Mutations to one F5Evidence must not affect another from the same input."""
        ev1 = norm(f4=FULL_F4)
        ev2 = norm(f4=FULL_F4)
        ev1.f4.catchlight_mismatch_flagged_frames.append("99:99:99")
        assert "99:99:99" not in ev2.f4.catchlight_mismatch_flagged_frames


# ===========================================================================
# All contract-defined F1 verdict values
# ===========================================================================

class TestF1AllowedVerdicts:
    @pytest.mark.parametrize("verdict_str,expected", [
        ("REAL", F1Verdict.REAL),
        ("KNOWN_SYNTHETIC", F1Verdict.KNOWN_SYNTHETIC),
        ("UNKNOWN_SYNTHETIC", F1Verdict.UNKNOWN_SYNTHETIC),
    ])
    def test_all_f1_verdicts(self, verdict_str, expected):
        data = {**FULL_F1, "verdict": verdict_str}
        ev = norm(f1=data)
        assert ev.f1.verdict is expected

    @pytest.mark.parametrize("level_str,expected", [
        ("LOW", Level.LOW),
        ("MED", Level.MED),
        ("HIGH", Level.HIGH),
    ])
    def test_all_f1_levels(self, level_str, expected):
        data = {**FULL_F1, "known_generator_similarity": level_str}
        ev = norm(f1=data)
        assert ev.f1.known_generator_similarity is expected


# ===========================================================================
# All contract-defined F2 pathway values
# ===========================================================================

class TestF2AllowedPathways:
    @pytest.mark.parametrize("pathway_str,expected", [
        ("DIRECT_GENUINE", F2Pathway.DIRECT_GENUINE),
        ("DIRECT_SYNTHETIC", F2Pathway.DIRECT_SYNTHETIC),
        ("REPLAYED_RECORDING", F2Pathway.REPLAYED_RECORDING),
    ])
    def test_all_f2_pathways(self, pathway_str, expected):
        data = {**FULL_F2, "pathway": pathway_str}
        ev = norm(f2=data)
        assert ev.f2.pathway is expected


# ===========================================================================
# All contract-defined F4 verdict values
# ===========================================================================

class TestF4AllowedVerdicts:
    def test_consistent(self):
        data = {**FULL_F4, "verdict": "CONSISTENT"}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.CONSISTENT

    def test_mismatch_detected(self):
        ev = norm(f4=FULL_F4)
        assert ev.f4.verdict is F4Verdict.MISMATCH_DETECTED

    def test_insufficient_data_maps_to_unknown(self):
        data = {**FULL_F4, "verdict": "INSUFFICIENT_DATA"}
        ev = norm(f4=data)
        assert ev.f4.verdict is F4Verdict.UNKNOWN
