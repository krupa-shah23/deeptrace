"""
Tests for src/p_fusion/fusion_engine.py.

Sources used:
    docs/contracts.md          — allowed field names and values
    src/p_fusion/fusion_engine.py — module under test

No invented evidence, thresholds, or rules.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from p_fusion.fusion_engine import main, run_fusion, run_fusion_from_files


# ---------------------------------------------------------------------------
# Canonical raw JSON inputs (from docs/contracts.md)
# ---------------------------------------------------------------------------

def _ai_synthesis_audio() -> dict[str, Any]:
    return {
        "f1": {
            "known_generator_similarity": "HIGH",
            "natural_speech_consistency": "LOW",
            "verdict": "UNKNOWN_SYNTHETIC",
            "confidence": 0.9,
        },
        "f2": {},  # Absent fields
    }


def _ai_synthesis_video() -> dict[str, Any]:
    return {
        "f3": {
            "match_found": False,
        },
        "f4": {
            "frames_with_usable_eyes": 20,
            "catchlight_mismatch_flagged_frames": ["00:00:04"],
            "mismatch_rate": 0.5,
            "verdict": "MISMATCH_DETECTED",
        },
    }


def _re_recorded_audio() -> dict[str, Any]:
    return {
        "f1": {
            "verdict": "KNOWN_SYNTHETIC",
            "confidence": 0.8,
        },
        "f2": {
            "pathway": "REPLAYED_RECORDING",
            "reverb_detected": True,
            "confidence": 0.85,
        },
    }


def _conventional_editing_video() -> dict[str, Any]:
    return {
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
    }


def _conventional_editing_audio() -> dict[str, Any]:
    return {
        "f1": {
            "verdict": "REAL",
            "confidence": 0.95,
        }
    }


# ---------------------------------------------------------------------------
# Test 1: Complete audio + video input & attribution outcomes
# ---------------------------------------------------------------------------

class TestAttributionOutcomes:
    
    def test_ai_synthesis(self):
        result = run_fusion(
            audio_data=_ai_synthesis_audio(),
            video_data=_ai_synthesis_video(),
        )
        assert result["attribution"] == "AI_SYNTHESIS"
        assert result["has_conflict"] is False

    def test_ai_generated_then_recorded(self):
        result = run_fusion(
            audio_data=_re_recorded_audio(),
            video_data=None,
        )
        assert result["attribution"] == "RE_RECORDED"
        assert result["has_conflict"] is False

    def test_conventional_editing(self):
        result = run_fusion(
            audio_data=_conventional_editing_audio(),
            video_data=_conventional_editing_video(),
        )
        assert result["attribution"] == "CONVENTIONAL_EDITING"
        assert result["has_conflict"] is False

    def test_inconclusive_empty_input(self):
        result = run_fusion(audio_data=None, video_data=None)
        assert result["attribution"] == "UNKNOWN"
        assert result["has_conflict"] is False


# ---------------------------------------------------------------------------
# Test 2: Missing F1/F2/F3/F4 and specific edge cases
# ---------------------------------------------------------------------------

class TestMissingAndEdgeCases:

    def test_missing_audio_handled_safely(self):
        result = run_fusion(audio_data=None, video_data=_ai_synthesis_video())
        assert result["evidence"]["f1"]["verdict"] == "UNKNOWN"
        assert result["evidence"]["f2"]["pathway"] == "UNKNOWN"
        assert result["attribution"] == "UNKNOWN"  # Missing F1 prevents AI_SYNTHESIS

    def test_missing_video_handled_safely(self):
        result = run_fusion(audio_data=_ai_synthesis_audio(), video_data=None)
        assert result["evidence"]["f3"]["match_found"] is None
        assert result["evidence"]["f4"]["verdict"] == "UNKNOWN"
        assert result["attribution"] == "UNKNOWN"  # Missing F4/F3 prevents AI_SYNTHESIS

    def test_f4_insufficient_data(self):
        video = _conventional_editing_video()
        video["f4"]["verdict"] = "INSUFFICIENT_DATA"
        result = run_fusion(
            audio_data=_conventional_editing_audio(),
            video_data=video,
        )
        # Normalizer should map INSUFFICIENT_DATA to UNKNOWN
        assert result["evidence"]["f4"]["verdict"] == "UNKNOWN"
        # And it should fail to attribute CONVENTIONAL_EDITING
        assert result["attribution"] == "UNKNOWN"

    def test_f3_match_found_false_preservation(self):
        result = run_fusion(
            audio_data=None,
            video_data=_ai_synthesis_video(),
        )
        # Should be preserved exactly as False, not None or True
        assert result["evidence"]["f3"]["match_found"] is False


# ---------------------------------------------------------------------------
# Test 3: Evidence and cause tree preservation
# ---------------------------------------------------------------------------

class TestPreservation:

    def test_evidence_preservation(self):
        result = run_fusion(
            audio_data=_conventional_editing_audio(),
            video_data=_conventional_editing_video(),
        )
        ev = result["evidence"]
        assert ev["f1"]["verdict"] == "REAL"
        assert ev["f3"]["match_found"] is True
        assert ev["f3"]["matched_seed_id"] == "seed_001"
        assert ev["f4"]["verdict"] == "CONSISTENT"
        
    def test_cause_tree_preservation(self):
        result = run_fusion(
            audio_data=_ai_synthesis_audio(),
            video_data=_ai_synthesis_video(),
        )
        tree = result["cause_tree"]
        assert tree["attribution"] == "AI_SYNTHESIS"
        assert len(tree["evidence_branches"]) == 4
        assert "AI_SYNTHESIS" in tree["reason"]
        
    def test_no_final_verdict_field(self):
        # We explicitly avoid FinalVerdict mapping, so it shouldn't be in the result
        result = run_fusion()
        assert "final_verdict" not in result


# ---------------------------------------------------------------------------
# Test 4: Determinism & JSON Serialisability
# ---------------------------------------------------------------------------

class TestDeterminismAndSerialisability:

    def test_deterministic_output(self):
        r1 = run_fusion(
            audio_data=_ai_synthesis_audio(),
            video_data=_ai_synthesis_video(),
        )
        r2 = run_fusion(
            audio_data=_ai_synthesis_audio(),
            video_data=_ai_synthesis_video(),
        )
        # Check identity isn't shared but content is identical
        assert r1 is not r2
        assert r1 == r2
        assert json.dumps(r1) == json.dumps(r2)

    @pytest.mark.parametrize("audio_func,video_func", [
        (_ai_synthesis_audio, _ai_synthesis_video),
        (_re_recorded_audio, lambda: None),
        (_conventional_editing_audio, _conventional_editing_video),
        (lambda: None, lambda: None),
    ])
    def test_json_serialisable(self, audio_func, video_func):
        result = run_fusion(
            audio_data=audio_func(),
            video_data=video_func(),
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["attribution"] == result["attribution"]


# ---------------------------------------------------------------------------
# Test 5: File loading and CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_json_files(tmp_path):
    audio_file = tmp_path / "audio.json"
    video_file = tmp_path / "video.json"
    
    audio_file.write_text(json.dumps(_ai_synthesis_audio()))
    video_file.write_text(json.dumps(_ai_synthesis_video()))
    
    return str(audio_file), str(video_file)


class TestFileAndCLI:

    def test_run_fusion_from_files(self, temp_json_files):
        audio_path, video_path = temp_json_files
        result = run_fusion_from_files(
            audio_json_path=audio_path,
            video_json_path=video_path,
        )
        assert result["attribution"] == "AI_SYNTHESIS"
        
    def test_run_fusion_from_files_missing_safely(self, temp_json_files):
        audio_path, _ = temp_json_files
        result = run_fusion_from_files(
            audio_json_path=audio_path,
            video_json_path=None,
        )
        assert result["attribution"] == "UNKNOWN"

    def test_run_fusion_from_files_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_fusion_from_files(audio_json_path="does_not_exist.json")

    def test_cli_main_success(self, temp_json_files, capsys):
        audio_path, video_path = temp_json_files
        
        # main returns 0 on success
        rc = main([
            "--audio-json", audio_path,
            "--video-json", video_path,
        ])
        assert rc == 0
        
        # check stdout
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["attribution"] == "AI_SYNTHESIS"

    def test_cli_main_pretty(self, temp_json_files, capsys):
        audio_path, video_path = temp_json_files
        
        rc = main([
            "--audio-json", audio_path,
            "--video-json", video_path,
            "--pretty"
        ])
        assert rc == 0
        
        captured = capsys.readouterr()
        # Pretty printed JSON will have newlines
        assert "\n" in captured.out
        output = json.loads(captured.out)
        assert output["attribution"] == "AI_SYNTHESIS"
        
    def test_cli_main_error_not_found(self, capsys):
        rc = main(["--audio-json", "does_not_exist.json"])
        assert rc == 1
        
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "not found" in captured.err.lower()
