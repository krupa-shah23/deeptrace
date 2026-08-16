"""
Tests for src/p_fusion/cause_tree.py.

Sources used:
    docs/contracts.md          — allowed field names and values
    src/p_fusion/schemas.py    — internal types
    src/p_fusion/attribution.py — produces the F5AttributionResult input
    src/p_fusion/cause_tree.py  — function under test

No invented evidence, thresholds, or rules.
Attribution results are obtained by calling attribution.attribute() with
valid contract-defined evidence — not by constructing F5AttributionResult
directly, so the cause tree tests always receive a realistic attribution.
"""

import json

import pytest

from p_fusion.attribution import attribute
from p_fusion.cause_tree import build_cause_tree
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
# Helpers
# ---------------------------------------------------------------------------

def _branch(tree: dict, feature: str) -> dict:
    """Return the evidence branch for the given feature label ("F1"–"F4")."""
    for b in tree["evidence_branches"]:
        if b["feature"] == feature:
            return b
    raise KeyError(f"No branch found for feature={feature!r}")


# ---------------------------------------------------------------------------
# Canonical evidence builders (mirrors test_attribution.py)
# ---------------------------------------------------------------------------

def _ai_synthesis_evidence() -> F5Evidence:
    return F5Evidence(
        f1=F1Evidence(
            known_generator_similarity=Level.HIGH,
            natural_speech_consistency=Level.LOW,
            verdict=F1Verdict.UNKNOWN_SYNTHETIC,
            confidence=0.9,
        ),
        f2=F2Evidence(),
        f3=F3Evidence(match_found=False),
        f4=F4Evidence(
            frames_with_usable_eyes=20,
            catchlight_mismatch_flagged_frames=["00:00:04"],
            mismatch_rate=0.5,
            verdict=F4Verdict.MISMATCH_DETECTED,
        ),
    )


def _re_recorded_evidence() -> F5Evidence:
    return F5Evidence(
        f1=F1Evidence(verdict=F1Verdict.KNOWN_SYNTHETIC, confidence=0.8),
        f2=F2Evidence(pathway=F2Pathway.REPLAYED_RECORDING, reverb_detected=True, confidence=0.85),
        f3=F3Evidence(),
        f4=F4Evidence(),
    )


def _conventional_editing_evidence() -> F5Evidence:
    return F5Evidence(
        f1=F1Evidence(verdict=F1Verdict.REAL, confidence=0.95),
        f2=F2Evidence(),
        f3=F3Evidence(
            match_found=True,
            matched_seed_id="seed_001",
            estimated_reencoding_stages=2,
            metadata=F3Metadata(encoder="H.264", creation_time="2024-01-01", resolution="1920x1080"),
        ),
        f4=F4Evidence(
            frames_with_usable_eyes=50,
            catchlight_mismatch_flagged_frames=[],
            mismatch_rate=0.0,
            verdict=F4Verdict.CONSISTENT,
        ),
    )


def _all_unknown_evidence() -> F5Evidence:
    return F5Evidence()


def _tree(ev: F5Evidence) -> dict:
    """Convenience: normalise evidence → attribute → build tree."""
    return build_cause_tree(ev, attribute(ev))


# ---------------------------------------------------------------------------
# Test 1 — AI_SYNTHESIS tree
# ---------------------------------------------------------------------------

class TestAISynthesisTree:

    def test_attribution_value(self):
        t = _tree(_ai_synthesis_evidence())
        assert t["attribution"] == "AI_SYNTHESIS"

    def test_has_conflict_false(self):
        t = _tree(_ai_synthesis_evidence())
        assert t["has_conflict"] is False

    def test_f1_role_contributing(self):
        t = _tree(_ai_synthesis_evidence())
        assert _branch(t, "F1")["role"] == "CONTRIBUTING"

    def test_f2_role_unknown_data(self):
        # F2 is absent for AI_SYNTHESIS evidence
        t = _tree(_ai_synthesis_evidence())
        assert _branch(t, "F2")["role"] == "UNKNOWN_DATA"

    def test_f3_role_contributing(self):
        t = _tree(_ai_synthesis_evidence())
        assert _branch(t, "F3")["role"] == "CONTRIBUTING"

    def test_f4_role_contributing(self):
        t = _tree(_ai_synthesis_evidence())
        assert _branch(t, "F4")["role"] == "CONTRIBUTING"

    def test_f3_notes_present_for_no_match(self):
        # match_found=False must always produce a note
        t = _tree(_ai_synthesis_evidence())
        assert len(_branch(t, "F3")["notes"]) > 0

    def test_f3_notes_say_no_source_match(self):
        t = _tree(_ai_synthesis_evidence())
        combined = " ".join(_branch(t, "F3")["notes"]).upper()
        assert "NO SOURCE MATCH" in combined or "NO_MATCH" in combined or "NO MATCH" in combined

    def test_f3_notes_do_not_say_manipulation(self):
        t = _tree(_ai_synthesis_evidence())
        combined = " ".join(_branch(t, "F3")["notes"]).lower()
        # The note must clarify it is NOT manipulation evidence
        assert "not" in combined and "manipulation" in combined

    def test_reason_is_non_empty_string(self):
        t = _tree(_ai_synthesis_evidence())
        assert isinstance(t["reason"], str) and len(t["reason"]) > 0

    def test_has_four_branches(self):
        t = _tree(_ai_synthesis_evidence())
        assert len(t["evidence_branches"]) == 4


# ---------------------------------------------------------------------------
# Test 2 — AI_GENERATED_THEN_RECORDED tree
# ---------------------------------------------------------------------------

class TestRERecordedTree:

    def test_attribution_value(self):
        t = _tree(_re_recorded_evidence())
        assert t["attribution"] == "RE_RECORDED"

    def test_has_conflict_false(self):
        t = _tree(_re_recorded_evidence())
        assert t["has_conflict"] is False

    def test_f1_role_contributing(self):
        t = _tree(_re_recorded_evidence())
        assert _branch(t, "F1")["role"] == "CONTRIBUTING"

    def test_f2_role_contributing(self):
        t = _tree(_re_recorded_evidence())
        assert _branch(t, "F2")["role"] == "CONTRIBUTING"

    def test_f3_role_unknown_data(self):
        # F3 data absent (match_found=None)
        t = _tree(_re_recorded_evidence())
        assert _branch(t, "F3")["role"] == "UNKNOWN_DATA"

    def test_f4_role_unknown_data(self):
        # F4 data absent (verdict=UNKNOWN)
        t = _tree(_re_recorded_evidence())
        assert _branch(t, "F4")["role"] == "UNKNOWN_DATA"

    def test_has_four_branches(self):
        assert len(_tree(_re_recorded_evidence())["evidence_branches"]) == 4


# ---------------------------------------------------------------------------
# Test 3 — CONVENTIONAL_EDITING tree
# ---------------------------------------------------------------------------

class TestConventionalEditingTree:

    def test_attribution_value(self):
        t = _tree(_conventional_editing_evidence())
        assert t["attribution"] == "CONVENTIONAL_EDITING"

    def test_has_conflict_false(self):
        t = _tree(_conventional_editing_evidence())
        assert t["has_conflict"] is False

    def test_f1_role_contributing(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F1")["role"] == "CONTRIBUTING"

    def test_f2_role_unknown_data(self):
        # F2 absent
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F2")["role"] == "UNKNOWN_DATA"

    def test_f3_role_contributing(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F3")["role"] == "CONTRIBUTING"

    def test_f4_role_contributing(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F4")["role"] == "CONTRIBUTING"

    def test_f3_fields_preserve_match_found_true(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F3")["fields"]["match_found"] is True

    def test_f3_fields_preserve_seed_id(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F3")["fields"]["matched_seed_id"] == "seed_001"

    def test_f3_fields_preserve_reencoding_stages(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F3")["fields"]["estimated_reencoding_stages"] == 2

    def test_f4_fields_preserve_verdict_consistent(self):
        t = _tree(_conventional_editing_evidence())
        assert _branch(t, "F4")["fields"]["verdict"] == "CONSISTENT"

    def test_has_four_branches(self):
        assert len(_tree(_conventional_editing_evidence())["evidence_branches"]) == 4


# ---------------------------------------------------------------------------
# Test 4 — INCONCLUSIVE tree
# ---------------------------------------------------------------------------

class TestInconclusiveTree:

    def test_attribution_is_unknown(self):
        t = _tree(_all_unknown_evidence())
        assert t["attribution"] == "UNKNOWN"

    def test_has_conflict_false(self):
        t = _tree(_all_unknown_evidence())
        assert t["has_conflict"] is False

    def test_all_branches_are_unknown_data(self):
        t = _tree(_all_unknown_evidence())
        for branch in t["evidence_branches"]:
            assert branch["role"] == "UNKNOWN_DATA", (
                f"Expected UNKNOWN_DATA for {branch['feature']}, got {branch['role']}"
            )

    def test_limitations_mention_inconclusive(self):
        t = _tree(_all_unknown_evidence())
        combined = " ".join(t["limitations"]).upper()
        assert "INCONCLUSIVE" in combined

    def test_reason_contains_inconclusive(self):
        t = _tree(_all_unknown_evidence())
        assert "INCONCLUSIVE" in t["reason"]

    def test_has_four_branches(self):
        assert len(_tree(_all_unknown_evidence())["evidence_branches"]) == 4


# ---------------------------------------------------------------------------
# Test 5 — Missing / UNKNOWN features
# ---------------------------------------------------------------------------

class TestUnknownFeatures:
    """Each feature with UNKNOWN/absent data must produce role=UNKNOWN_DATA."""

    def test_unknown_f1_produces_unknown_data_role(self):
        ev = F5Evidence(f1=F1Evidence(verdict=F1Verdict.UNKNOWN))
        t = _tree(ev)
        assert _branch(t, "F1")["role"] == "UNKNOWN_DATA"

    def test_unknown_f2_produces_unknown_data_role(self):
        ev = F5Evidence(f2=F2Evidence(pathway=F2Pathway.UNKNOWN))
        t = _tree(ev)
        assert _branch(t, "F2")["role"] == "UNKNOWN_DATA"

    def test_f3_match_found_none_produces_unknown_data_role(self):
        ev = F5Evidence(f3=F3Evidence(match_found=None))
        t = _tree(ev)
        assert _branch(t, "F3")["role"] == "UNKNOWN_DATA"

    def test_unknown_f4_produces_unknown_data_role(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.UNKNOWN))
        t = _tree(ev)
        assert _branch(t, "F4")["role"] == "UNKNOWN_DATA"

    def test_all_unknown_all_branches_unknown_data(self):
        t = _tree(_all_unknown_evidence())
        roles = [b["role"] for b in t["evidence_branches"]]
        assert all(r == "UNKNOWN_DATA" for r in roles)

    def test_limitations_list_unknown_features(self):
        t = _tree(_all_unknown_evidence())
        combined = " ".join(t["limitations"])
        # At minimum, features with no data should be mentioned
        assert "F1" in combined or "F2" in combined or "F3" in combined or "F4" in combined


# ---------------------------------------------------------------------------
# Test 6 — F3 match_found=False is NO_MATCH, never manipulation evidence
# ---------------------------------------------------------------------------

class TestF3NoMatchRepresentation:

    def test_match_found_false_stored_in_fields_as_false(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        assert _branch(t, "F3")["fields"]["match_found"] is False

    def test_match_found_false_produces_a_note(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        assert len(_branch(t, "F3")["notes"]) > 0

    def test_match_found_false_note_says_no_source_match(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        combined = " ".join(_branch(t, "F3")["notes"]).upper()
        assert "NO SOURCE MATCH" in combined or "NO MATCH" in combined

    def test_match_found_false_note_says_not_manipulation(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        combined = " ".join(_branch(t, "F3")["notes"]).lower()
        assert "not" in combined and "manipulation" in combined

    def test_match_found_false_note_says_not_independently_proof(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        combined = " ".join(_branch(t, "F3")["notes"]).lower()
        # Note should explain F3 alone is insufficient
        assert "alone" in combined or "independent" in combined or "combination" in combined

    def test_match_found_none_produces_absent_note(self):
        ev = F5Evidence(f3=F3Evidence(match_found=None))
        t = _tree(ev)
        notes = _branch(t, "F3")["notes"]
        assert len(notes) > 0
        combined = " ".join(notes).lower()
        assert "absent" in combined or "unknown" in combined

    def test_match_found_none_role_is_unknown_data(self):
        ev = F5Evidence(f3=F3Evidence(match_found=None))
        t = _tree(ev)
        assert _branch(t, "F3")["role"] == "UNKNOWN_DATA"

    def test_match_found_false_fields_do_not_contain_manipulation_key(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        fields = _branch(t, "F3")["fields"]
        for key in fields:
            assert "manipulation" not in key.lower(), (
                f"Unexpected manipulation-related key in F3 fields: {key!r}"
            )

    def test_match_found_false_attribution_is_not_ai_synthesis_with_only_f3(self):
        # F3 no-match alone must not produce any non-UNKNOWN cause
        ev = F5Evidence(f3=F3Evidence(match_found=False))
        t = _tree(ev)
        assert t["attribution"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Test 7 — F4 INSUFFICIENT_DATA / UNKNOWN is represented as UNKNOWN_DATA
# ---------------------------------------------------------------------------

class TestF4UnknownRepresentation:

    def test_f4_unknown_verdict_role_is_unknown_data(self):
        # UNKNOWN represents both absent data and post-normalisation INSUFFICIENT_DATA
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.UNKNOWN))
        t = _tree(ev)
        assert _branch(t, "F4")["role"] == "UNKNOWN_DATA"

    def test_f4_unknown_produces_a_note(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.UNKNOWN))
        t = _tree(ev)
        assert len(_branch(t, "F4")["notes"]) > 0

    def test_f4_unknown_note_says_cannot_be_consistent(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.UNKNOWN))
        t = _tree(ev)
        combined = " ".join(_branch(t, "F4")["notes"]).upper()
        assert "CONSISTENT" in combined

    def test_f4_unknown_note_says_unknown_or_insufficient(self):
        ev = F5Evidence(f4=F4Evidence(verdict=F4Verdict.UNKNOWN))
        t = _tree(ev)
        combined = " ".join(_branch(t, "F4")["notes"]).upper()
        assert "UNKNOWN" in combined or "INSUFFICIENT" in combined

    def test_f4_consistent_is_not_unknown_data(self):
        # CONSISTENT is a definitive value and must not produce UNKNOWN_DATA role
        # (it would be CONTRIBUTING in CONVENTIONAL_EDITING or NEUTRAL otherwise)
        ev = _conventional_editing_evidence()
        t = _tree(ev)
        assert _branch(t, "F4")["role"] != "UNKNOWN_DATA"

    def test_f4_mismatch_is_not_unknown_data(self):
        ev = _ai_synthesis_evidence()
        t = _tree(ev)
        assert _branch(t, "F4")["role"] != "UNKNOWN_DATA"


# ---------------------------------------------------------------------------
# Test 8 — Every F1–F4 branch identifies its source feature
# ---------------------------------------------------------------------------

class TestBranchStructure:

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_exactly_four_branches(self, ev):
        assert len(_tree(ev)["evidence_branches"]) == 4

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_all_four_features_represented(self, ev):
        features = {b["feature"] for b in _tree(ev)["evidence_branches"]}
        assert features == {"F1", "F2", "F3", "F4"}

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_all_branches_have_label(self, ev):
        for b in _tree(ev)["evidence_branches"]:
            assert isinstance(b.get("label"), str) and len(b["label"]) > 0

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_all_branches_have_fields_dict(self, ev):
        for b in _tree(ev)["evidence_branches"]:
            assert isinstance(b.get("fields"), dict)

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_all_branches_have_notes_list(self, ev):
        for b in _tree(ev)["evidence_branches"]:
            assert isinstance(b.get("notes"), list)

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_all_branches_have_role(self, ev):
        valid_roles = {"CONTRIBUTING", "NEUTRAL", "UNKNOWN_DATA"}
        for b in _tree(ev)["evidence_branches"]:
            assert b.get("role") in valid_roles, (
                f"Unexpected role {b.get('role')!r} for feature {b.get('feature')!r}"
            )

    def test_f1_fields_contain_verdict(self):
        t = _tree(_ai_synthesis_evidence())
        assert "verdict" in _branch(t, "F1")["fields"]

    def test_f2_fields_contain_pathway(self):
        t = _tree(_re_recorded_evidence())
        assert "pathway" in _branch(t, "F2")["fields"]

    def test_f3_fields_contain_match_found(self):
        t = _tree(_ai_synthesis_evidence())
        assert "match_found" in _branch(t, "F3")["fields"]

    def test_f4_fields_contain_verdict(self):
        t = _tree(_ai_synthesis_evidence())
        assert "verdict" in _branch(t, "F4")["fields"]

    def test_f4_fields_contain_mismatch_rate(self):
        t = _tree(_ai_synthesis_evidence())
        assert "mismatch_rate" in _branch(t, "F4")["fields"]

    def test_f4_fields_contain_frames_with_usable_eyes(self):
        t = _tree(_ai_synthesis_evidence())
        assert "frames_with_usable_eyes" in _branch(t, "F4")["fields"]

    def test_f4_fields_contain_catchlight_flagged_frames(self):
        t = _tree(_ai_synthesis_evidence())
        assert "catchlight_mismatch_flagged_frames" in _branch(t, "F4")["fields"]

    def test_f3_fields_contain_estimated_reencoding_stages(self):
        t = _tree(_conventional_editing_evidence())
        assert "estimated_reencoding_stages" in _branch(t, "F3")["fields"]

    def test_f3_fields_contain_metadata(self):
        t = _tree(_conventional_editing_evidence())
        assert "metadata" in _branch(t, "F3")["fields"]


# ---------------------------------------------------------------------------
# Test 9 — Attribution is preserved exactly
# ---------------------------------------------------------------------------

class TestAttributionPreservation:

    @pytest.mark.parametrize("ev,expected_cause", [
        (_ai_synthesis_evidence(),        "AI_SYNTHESIS"),
        (_re_recorded_evidence(),         "RE_RECORDED"),
        (_conventional_editing_evidence(), "CONVENTIONAL_EDITING"),
        (_all_unknown_evidence(),         "UNKNOWN"),
    ])
    def test_attribution_value_matches(self, ev, expected_cause):
        t = _tree(ev)
        assert t["attribution"] == expected_cause

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_has_conflict_matches_attribution_result(self, ev):
        attr = attribute(ev)
        t = build_cause_tree(ev, attr)
        assert t["has_conflict"] == attr.conflicts.has_conflict

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_reason_matches_attribution_details(self, ev):
        attr = attribute(ev)
        t = build_cause_tree(ev, attr)
        assert t["reason"] == attr.conflicts.details


# ---------------------------------------------------------------------------
# Test 10 — Output is JSON serialisable
# ---------------------------------------------------------------------------

class TestJSONSerialisation:

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_tree_is_json_serialisable(self, ev):
        t = _tree(ev)
        # Must not raise
        result = json.dumps(t)
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_json_round_trip_preserves_attribution(self, ev):
        t = _tree(ev)
        reloaded = json.loads(json.dumps(t))
        assert reloaded["attribution"] == t["attribution"]

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_json_round_trip_preserves_has_conflict(self, ev):
        t = _tree(ev)
        reloaded = json.loads(json.dumps(t))
        assert reloaded["has_conflict"] == t["has_conflict"]

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_json_round_trip_preserves_branch_count(self, ev):
        t = _tree(ev)
        reloaded = json.loads(json.dumps(t))
        assert len(reloaded["evidence_branches"]) == len(t["evidence_branches"])

    def test_f4_catchlight_list_is_json_serialisable(self):
        ev = _ai_synthesis_evidence()
        t = _tree(ev)
        f4_fields = _branch(t, "F4")["fields"]
        # Must be a list, not a dataclass list field
        assert isinstance(f4_fields["catchlight_mismatch_flagged_frames"], list)
        json.dumps(f4_fields["catchlight_mismatch_flagged_frames"])

    def test_f3_metadata_dict_is_json_serialisable(self):
        t = _tree(_conventional_editing_evidence())
        meta = _branch(t, "F3")["fields"]["metadata"]
        assert isinstance(meta, dict)
        json.dumps(meta)

    def test_f3_metadata_none_is_json_serialisable(self):
        ev = F5Evidence(f3=F3Evidence(match_found=False))  # no metadata
        t = _tree(ev)
        meta = _branch(t, "F3")["fields"]["metadata"]
        assert meta is None
        json.dumps({"metadata": meta})


# ---------------------------------------------------------------------------
# Test 11 — Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_same_evidence_same_attribution_in_tree(self, ev):
        attr = attribute(ev)
        t1 = build_cause_tree(ev, attr)
        t2 = build_cause_tree(ev, attr)
        assert t1["attribution"] == t2["attribution"]

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_same_evidence_same_reason_in_tree(self, ev):
        attr = attribute(ev)
        t1 = build_cause_tree(ev, attr)
        t2 = build_cause_tree(ev, attr)
        assert t1["reason"] == t2["reason"]

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_same_evidence_same_branch_roles(self, ev):
        attr = attribute(ev)
        t1 = build_cause_tree(ev, attr)
        t2 = build_cause_tree(ev, attr)
        roles1 = [(b["feature"], b["role"]) for b in t1["evidence_branches"]]
        roles2 = [(b["feature"], b["role"]) for b in t2["evidence_branches"]]
        assert roles1 == roles2

    def test_two_calls_produce_independent_dicts(self):
        ev = _ai_synthesis_evidence()
        attr = attribute(ev)
        t1 = build_cause_tree(ev, attr)
        t2 = build_cause_tree(ev, attr)
        assert t1 is not t2


# ---------------------------------------------------------------------------
# Test 12 — cause_tree does not change or override attribution
# ---------------------------------------------------------------------------

class TestNoOverride:

    @pytest.mark.parametrize("ev,expected_cause", [
        (_ai_synthesis_evidence(),        AttributionCause.AI_SYNTHESIS),
        (_re_recorded_evidence(),         AttributionCause.RE_RECORDED),
        (_conventional_editing_evidence(), AttributionCause.CONVENTIONAL_EDITING),
        (_all_unknown_evidence(),         AttributionCause.UNKNOWN),
    ])
    def test_cause_tree_attribution_string_matches_attribution_cause(self, ev, expected_cause):
        attr = attribute(ev)
        t = build_cause_tree(ev, attr)
        assert t["attribution"] == expected_cause.value

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_attribution_object_cause_unchanged_after_tree_build(self, ev):
        attr = attribute(ev)
        original_cause = attr.cause
        build_cause_tree(ev, attr)
        assert attr.cause is original_cause

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_attribution_conflict_unchanged_after_tree_build(self, ev):
        attr = attribute(ev)
        original_conflict = attr.conflicts.has_conflict
        build_cause_tree(ev, attr)
        assert attr.conflicts.has_conflict == original_conflict

    @pytest.mark.parametrize("ev", [
        _ai_synthesis_evidence(),
        _re_recorded_evidence(),
        _conventional_editing_evidence(),
        _all_unknown_evidence(),
    ])
    def test_tree_attribution_equals_attribution_result_not_re_derived(self, ev):
        """Tree must carry the attribution.cause value, not independently compute it."""
        attr = attribute(ev)
        t = build_cause_tree(ev, attr)
        # The tree attribution must exactly match what attribution.py produced
        assert t["attribution"] == attr.cause.value
