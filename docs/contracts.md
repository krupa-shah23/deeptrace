# Integration Contracts

This document defines the canonical JSON integration contracts between the P-Video, P-Audio, and P-Fusion modules.

## F1 Zero-Day Audio

Detection of unknown/zero-day synthetic audio generators.

```json
{
  "known_generator_similarity": "LOW|MED|HIGH",
  "natural_speech_consistency": "LOW|MED|HIGH",
  "verdict": "REAL|KNOWN_SYNTHETIC|UNKNOWN_SYNTHETIC",
  "confidence": 0.0
}
```

*   `known_generator_similarity` (String): Indicates the similarity to known synthetic generators. Allowed values: `LOW`, `MED`, `HIGH`.
*   `natural_speech_consistency` (String): Indicates the consistency with natural speech patterns. Allowed values: `LOW`, `MED`, `HIGH`.
*   `verdict` (String): The final classification. Allowed values: `REAL`, `KNOWN_SYNTHETIC`, `UNKNOWN_SYNTHETIC`.
*   `confidence` (Float): A value between 0.0 and 1.0 representing the confidence in the verdict. Missing data implies `UNKNOWN` confidence.

## F2 Replay Detection

Detection of physical-world re-recording or replay attacks.

```json
{
  "pathway": "DIRECT_GENUINE|DIRECT_SYNTHETIC|REPLAYED_RECORDING",
  "reverb_detected": false,
  "confidence": 0.0
}
```

*   `pathway` (String): The estimated recording pathway. Allowed values: `DIRECT_GENUINE`, `DIRECT_SYNTHETIC`, `REPLAYED_RECORDING`.
*   `reverb_detected` (Boolean): True if artificial or environmental reverberation indicative of replay is detected.
*   `confidence` (Float): A value between 0.0 and 1.0 representing the confidence. Missing data implies `UNKNOWN` confidence.

## F3 Source/Propagation

Forensic tracking of the video source and its propagation history.

```json
{
  "match_found": false,
  "matched_seed_id": null,
  "estimated_reencoding_stages": 0,
  "metadata": {
    "encoder": "...",
    "creation_time": "...",
    "resolution": "..."
  }
}
```

*   `match_found` (Boolean): True if a match was found in the database. **Note:** `match_found=false` means NO MATCH, not proof of manipulation.
*   `matched_seed_id` (String | null): The ID of the matched seed clip, or null if no match was found.
*   `estimated_reencoding_stages` (Integer): A heuristic estimate of the number of re-encoding stages (0-N). Not an exact reconstruction count.
*   `metadata` (Object): Container for extracted metadata.
    *   `encoder` (String): The video encoder used.
    *   `creation_time` (String): The creation timestamp.
    *   `resolution` (String): The video resolution.

## F4 Eye Reflection

Analysis of eye reflection consistency across frames.

```json
{
  "frames_with_usable_eyes": 0,
  "catchlight_mismatch_flagged_frames": ["00:00:04"],
  "mismatch_rate": 0.0,
  "verdict": "CONSISTENT|MISMATCH_DETECTED|INSUFFICIENT_DATA"
}
```

*   `frames_with_usable_eyes` (Integer): Number of frames where eyes were clear enough for analysis (0-100).
*   `catchlight_mismatch_flagged_frames` (Array of Strings): Timestamps of frames where mismatches were detected.
*   `mismatch_rate` (Float): The proportion of usable frames showing mismatches (0.0 to 1.0).
*   `verdict` (String): The final evaluation. Allowed values: `CONSISTENT`, `MISMATCH_DETECTED`, `INSUFFICIENT_DATA`. **Note:** `INSUFFICIENT_DATA` must remain distinguishable from `CONSISTENT`.
