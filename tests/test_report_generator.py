"""
Tests for src/p_fusion/report_generator.py.

Sources used:
    docs/contracts.md           — allowed field names and values
    src/p_fusion/fusion_engine.py — produces the F5 result dict
    src/p_fusion/report_generator.py — module under test

No invented evidence, thresholds, or rules.
"""

import copy
import json
from typing import Any

import pytest

from p_fusion.fusion_engine import run_fusion
from p_fusion.report_generator import generate_report


# ---------------------------------------------------------------------------
# Canonical raw JSON inputs (from docs/contracts.md)
# ---------------------------------------------------------------------------

def _ai_synthesis_result() -> dict[str, Any]:
    return run_fusion(
        audio_data={
            "f1": {
                "known_generator_similarity": "HIGH",
                "natural_speech_consistency": "LOW",
                "verdict": "UNKNOWN_SYNTHETIC",
                "confidence": 0.9,
            },
            "f2": {},
        },
        video_data={
            "f3": {"match_found": False},
            "f4": {
                "frames_with_usable_eyes": 20,
                "catchlight_mismatch_flagged_frames": ["00:00:04"],
                "mismatch_rate": 0.5,
                "verdict": "MISMATCH_DETECTED",
            },
        },
    )


def _re_recorded_result() -> dict[str, Any]:
    return run_fusion(
        audio_data={
            "f1": {"verdict": "KNOWN_SYNTHETIC", "confidence": 0.8},
            "f2": {"pathway": "REPLAYED_RECORDING", "reverb_detected": True, "confidence": 0.85},
        },
        video_data=None,
    )


def _conventional_editing_result() -> dict[str, Any]:
    return run_fusion(
        audio_data={
            "f1": {"verdict": "REAL", "confidence": 0.95}
        },
        video_data={
            "f3": {
                "match_found": True,
                "matched_seed_id": "seed_001",
                "estimated_reencoding_stages": 2,
                "metadata": {
                    "encoder": "H.264",
                    "creation_time": "2024-01-01T00:00:00Z",
                    "resolution": "1920x1080",
                },
            },
            "f4": {
                "frames_with_usable_eyes": 50,
                "catchlight_mismatch_flagged_frames": [],
                "mismatch_rate": 0.0,
                "verdict": "CONSISTENT",
            },
        },
    )


def _inconclusive_result() -> dict[str, Any]:
    return run_fusion(audio_data=None, video_data=None)


# ---------------------------------------------------------------------------
# Test 1: Complete Reports by Category
# ---------------------------------------------------------------------------

class TestReportCategories:

    def test_ai_synthesis_report(self):
        report = generate_report(_ai_synthesis_result())
        assert "Result**: AI_SYNTHESIS" in report
        assert "90.0%" in report  # F1 confidence
        assert "UNKNOWN_SYNTHETIC" in report
        assert "MISMATCH_DETECTED" in report
        assert "50.0%" in report  # F4 mismatch rate

    def test_re_recorded_report(self):
        report = generate_report(_re_recorded_result())
        assert "Result**: RE_RECORDED" in report
        assert "KNOWN_SYNTHETIC" in report
        assert "REPLAYED_RECORDING" in report
        assert "85.0%" in report  # F2 confidence

    def test_conventional_editing_report(self):
        report = generate_report(_conventional_editing_result())
        assert "Result**: CONVENTIONAL_EDITING" in report
        assert "seed_001" in report
        assert "Estimated Re-encoding Stages**: 2" in report
        assert "H.264" in report
        assert "CONSISTENT" in report

    def test_inconclusive_report(self):
        report = generate_report(_inconclusive_result())
        assert "Result**: UNKNOWN" in report
        assert "Status: UNKNOWN (No definitive data)" in report


# ---------------------------------------------------------------------------
# Test 2: UNKNOWN/INSUFFICIENT evidence presentation
# ---------------------------------------------------------------------------

class TestUnknownEvidencePresentation:

    def test_missing_f1_shows_unknown_status(self):
        report = generate_report(_re_recorded_result())  # F3/F4 are missing here
        assert "### F3: Source/Propagation\n*Status: UNKNOWN (No definitive data)*" in report
        assert "### F4: Eye Reflection Catchlights\n- **Verdict**: UNKNOWN" in report

    def test_f4_insufficient_data_note_present_when_unknown(self):
        report = generate_report(_inconclusive_result())
        assert "INSUFFICIENT_DATA" in report
        assert "cannot be interpreted as CONSISTENT" in report

    def test_f4_insufficient_data_note_absent_when_consistent(self):
        report = generate_report(_conventional_editing_result())
        assert "cannot be interpreted as CONSISTENT" not in report


# ---------------------------------------------------------------------------
# Test 3: F3 NO_MATCH constraints
# ---------------------------------------------------------------------------

class TestF3NoMatchDescription:

    def test_f3_match_found_false_preserves_not_proof_of_manipulation_note(self):
        report = generate_report(_ai_synthesis_result())
        # Should explicitly state it is NOT proof of manipulation
        assert "NOT proof of manipulation" in report
        assert "Match Found**: False" in report

    def test_f3_match_found_true_omits_manipulation_note(self):
        report = generate_report(_conventional_editing_result())
        assert "NOT proof of manipulation" not in report
        assert "Match Found**: True" in report


# ---------------------------------------------------------------------------
# Test 4: Cause Tree and Limitations inclusion
# ---------------------------------------------------------------------------

class TestCauseTreeAndLimitations:

    def test_cause_tree_branches_rendered(self):
        report = generate_report(_ai_synthesis_result())
        assert "Cause Tree (Decision Path)" in report
        assert "F1** (Zero-Day Audio)" in report
        assert "[CONTRIBUTING]" in report
        assert "[UNKNOWN_DATA]" in report

    def test_limitations_included(self):
        report = generate_report(_ai_synthesis_result())
        assert "Limitations and Constraints" in report
        assert "This cause tree is explanatory only" in report

    def test_conflict_warning_included_if_conflict_present(self):
        # We simulate a conflict manually since generating one legally requires contradictory inputs
        result = _inconclusive_result()
        result["has_conflict"] = True
        report = generate_report(result)
        assert "ATTENTION: Conflicting evidence detected" in report


# ---------------------------------------------------------------------------
# Test 5: Metadata & Purity
# ---------------------------------------------------------------------------

class TestMetadataAndPurity:

    def test_case_info_rendered(self):
        report = generate_report(_ai_synthesis_result(), case_info={"sha256": "abcdef123", "input_file": "test.mp4"})
        assert "Sha256**: abcdef123" in report
        assert "Input File**: test.mp4" in report

    def test_no_fabricated_measurements(self):
        report = generate_report(_inconclusive_result())
        # Should not invent thresholds
        assert "0.0%" not in report
        assert "100.0%" not in report
        assert "threshold" not in report.lower()

    def test_deterministic_output(self):
        result = _ai_synthesis_result()
        r1 = generate_report(result)
        r2 = generate_report(result)
        assert r1 == r2

    def test_generation_does_not_mutate_f5_result(self):
        result = _ai_synthesis_result()
        result_copy = copy.deepcopy(result)
        
        generate_report(result)
        
        # Verify exact structural match after execution
        assert json.dumps(result) == json.dumps(result_copy)
