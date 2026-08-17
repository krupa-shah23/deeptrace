"""
P-Fusion report generator.

Generates a human-readable, deterministic markdown report from the final F5
fusion dict produced by fusion_engine.py.

Constraints (strictly enforced):
- Only reports evidence present in the provided dict.
- Does not run detection or modify attribution.
- Does not invent probabilities, thresholds, or evidence.
- Explains semantic constraints (e.g., F3 no-match, F4 insufficient data).
- Standard library only; no new dependencies.
- No imports from p_audio or p_video.
"""

from typing import Any, Optional


def _format_confidence(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def generate_report(f5_result: dict[str, Any], case_info: Optional[dict[str, str]] = None) -> str:
    """
    Generate a formatted forensic report from the F5 fusion result.

    Args:
        f5_result: The JSON-serialisable dict produced by fusion_engine.py.
        case_info: Optional metadata about the case (e.g. filename, sha256, date).

    Returns:
        A Markdown-formatted report string.
    """
    lines = []

    lines.append("# DeepTrace Forensic Attribution Report")
    lines.append("")

    # --- 1. Case Information ---
    if case_info:
        lines.append("## Case Information")
        for key, val in case_info.items():
            # e.g., "Input File", "SHA-256", "Date"
            # Format nicely
            label = str(key).replace("_", " ").title()
            lines.append(f"- **{label}**: {val}")
        lines.append("")

    # --- 2. Final Attribution & Reasoning ---
    attribution = f5_result.get("attribution", "UNKNOWN")
    reason = f5_result.get("reason", "No reason provided.")
    has_conflict = f5_result.get("has_conflict", False)

    lines.append("## Final Attribution")
    
    # Highlight conflict if present
    if has_conflict:
        lines.append("**ATTENTION: Conflicting evidence detected.**")
        lines.append("")

    lines.append(f"**Result**: {attribution}")
    lines.append("")
    lines.append("### Reasoning")
    lines.append(f"{reason}")
    lines.append("")

    # --- 3. Evidence Breakdown ---
    lines.append("## Evidence Breakdown")
    lines.append("")
    
    ev = f5_result.get("evidence", {})

    # F1 (Zero-Day Audio)
    f1 = ev.get("f1", {})
    f1_verdict = f1.get("verdict", "UNKNOWN")
    lines.append("### F1: Zero-Day Audio")
    if f1_verdict == "UNKNOWN":
        lines.append("*Status: UNKNOWN (No definitive data)*")
    else:
        conf = _format_confidence(f1.get("confidence"))
        lines.append(f"- **Verdict**: {f1_verdict}")
        lines.append(f"- **Confidence**: {conf}")
        lines.append(f"- **Generator Similarity**: {f1.get('known_generator_similarity', 'UNKNOWN')}")
        lines.append(f"- **Natural Speech Consistency**: {f1.get('natural_speech_consistency', 'UNKNOWN')}")
    lines.append("")

    # F2 (Replay Detection)
    f2 = ev.get("f2", {})
    f2_pathway = f2.get("pathway", "UNKNOWN")
    lines.append("### F2: Replay Detection")
    if f2_pathway == "UNKNOWN":
        lines.append("*Status: UNKNOWN (No definitive data)*")
    else:
        conf = _format_confidence(f2.get("confidence"))
        lines.append(f"- **Pathway**: {f2_pathway}")
        lines.append(f"- **Confidence**: {conf}")
        lines.append(f"- **Reverb Detected**: {f2.get('reverb_detected')}")
    lines.append("")

    # F3 (Source/Propagation)
    f3 = ev.get("f3", {})
    match_found = f3.get("match_found")
    lines.append("### F3: Source/Propagation")
    if match_found is None:
        lines.append("*Status: UNKNOWN (No definitive data)*")
    else:
        lines.append(f"- **Match Found**: {match_found}")
        if match_found is False:
            lines.append("  > *Note: Absence of a known source seed (match_found=False) is NOT proof of manipulation. It only indicates novel content.*")
        else:
            lines.append(f"- **Matched Seed ID**: {f3.get('matched_seed_id')}")
            
        re_stages = f3.get("estimated_reencoding_stages")
        if re_stages is not None:
            lines.append(f"- **Estimated Re-encoding Stages**: {re_stages} (heuristic)")
            
        meta = f3.get("metadata")
        if meta:
            lines.append("- **Metadata**:")
            lines.append(f"  - Encoder: {meta.get('encoder')}")
            lines.append(f"  - Creation Time: {meta.get('creation_time')}")
            lines.append(f"  - Resolution: {meta.get('resolution')}")
    lines.append("")

    # F4 (Eye Reflection)
    f4 = ev.get("f4", {})
    f4_verdict = f4.get("verdict", "UNKNOWN")
    lines.append("### F4: Eye Reflection Catchlights")
    
    # We still print stats even if verdict is UNKNOWN, because it might be INSUFFICIENT_DATA mapped to UNKNOWN.
    # The normaliser preserves the numerics so the report can document why.
    usable = f4.get("frames_with_usable_eyes")
    if usable is not None:
        lines.append(f"- **Frames with Usable Eyes**: {usable}")
        
    rate = f4.get("mismatch_rate")
    if rate is not None:
        lines.append(f"- **Mismatch Rate**: {_format_confidence(rate)}")

    flagged = f4.get("catchlight_mismatch_flagged_frames", [])
    if flagged:
        lines.append(f"- **Flagged Frames**: {', '.join(flagged)}")

    lines.append(f"- **Verdict**: {f4_verdict}")
    if f4_verdict == "UNKNOWN":
        lines.append("  > *Note: UNKNOWN verdict means absent data or INSUFFICIENT_DATA (detector could not gather enough usable frames). It cannot be interpreted as CONSISTENT.*")

    lines.append("")

    # --- 4. Cause Tree ---
    lines.append("## Cause Tree (Decision Path)")
    lines.append("")
    tree = f5_result.get("cause_tree", {})
    branches = tree.get("evidence_branches", [])
    
    if not branches:
        lines.append("*Cause tree unavailable.*")
    else:
        for b in branches:
            feature = b.get("feature", "")
            label = b.get("label", "")
            role = b.get("role", "UNKNOWN")
            
            # Simple format: - F1 (Zero-Day Audio) [CONTRIBUTING]
            lines.append(f"- **{feature}** ({label}) — [{role}]")
            
            # If there are specific notes for this branch, list them
            for note in b.get("notes", []):
                lines.append(f"  - *{note}*")
    
    lines.append("")

    # --- 5. Limitations ---
    lines.append("## Limitations and Constraints")
    lines.append("")
    limitations = f5_result.get("limitations", [])
    if limitations:
        for lim in limitations:
            lines.append(f"- {lim}")
    else:
        lines.append("*No specific limitations documented.*")

    return "\n".join(lines)
