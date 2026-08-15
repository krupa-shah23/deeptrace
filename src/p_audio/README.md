# DeepTrace P-Audio Forensic Module

The **P-Audio** module provides forensic audio authentication for DeepTrace (SIH 2026 Hackathon). It analyzes suspect audio tracks (or audio extracted from suspect video containers) to produce explainable evidence across two primary detection engines:

- **F1 (Zero-Day / Unknown-Generator Detection)**: Flags synthetic/AI-generated speech without relying on a closed-set list of known generators.
- **F2 (Replay / Physical-World Re-recording Detection)**: Detects physical acoustic playback through speakers, room reverberation, and re-recording via microphones.

---

## Architecture

```
                          Suspect Audio / Video Clip
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   Mono Convert              16 kHz Resample
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                             Speech VAD & Trimming
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      F1: Zero-Day Novelty Detector         F2: Replay Pathway Detector
       (Pretrained + One-Class SVM)           (T60 Reverb, Noise Floor, Comb)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                           Structured JSON Contract
                                      │
                                      ▼
                        P-Fusion Attribution Engine
```

---

## Technical Approach

### F1 — Zero-Day / Unknown-Generator Synthetic Audio Detection
F1 combines two complementary signals:
1. **Pretrained Anti-Spoofing Model**: Computes similarity score against known synthetic vocoder / TTS generation patterns using a HuggingFace audio model or robust acoustic feature embedding baseline.
2. **One-Class SVM Novelty Detector**: Trained **strictly on genuine human speech** features (MFCCs, LFCCs, PYIN F0 pitch statistics, micro-tremor jitter, shimmer, and STFT phase continuity). Audio that strays outside the learned real-speech distribution is flagged as an anomaly.

**Verdicts produced by F1:**
- `REAL`: High natural speech consistency, low known generator similarity.
- `KNOWN_SYNTHETIC`: High known generator similarity.
- `UNKNOWN_SYNTHETIC`: Low natural speech consistency, low known generator similarity (Zero-Day detected!).
- `INCONCLUSIVE`: Speech clip too short (< 2.0s usable speech) or contradictory evidence.

### F2 — Replay / Physical-World Re-recording Detection
F2 measures acoustic properties introduced when audio is played through a physical speaker into a room and re-recorded with a microphone:
- **T60 Reverberation Time Proxy**: Schroeder backward integration energy decay curve slope.
- **Noise Floor & SNR**: Minimum 10th-percentile spectral frame energy vs active speech energy.
- **Comb Filtering Index**: Autocorrelation of magnitude spectrum peaks to detect periodic acoustic reflections.
- **Spectral Flatness & Rolloff**: High-frequency acoustic attenuation.

**Pathways produced by F2:**
- `DIRECT_GENUINE`: Clean direct digital/mic recording of real voice.
- `DIRECT_SYNTHETIC`: Direct digital output of AI voice generator without room playback.
- `REPLAYED_RECORDING`: Audio played through speaker in a physical room and re-recorded.
- `INCONCLUSIVE`: Audio clip unusable or insufficient evidence.

---

## Usage

### 1. Synchronize Dependencies
```powershell
uv sync
```

### 2. Train F1 One-Class SVM Baseline
```powershell
uv run python src/p_audio/train_f1.py --data-dir data/audio/real --out-model models/f1_one_class_svm.joblib
```
*(If `data/audio/real` is empty, the training script automatically generates clean synthetic baseline speech samples to fit the One-Class SVM baseline out-of-the-box.)*

### 3. Run Complete P-Audio Pipeline
```powershell
uv run python scripts/run_audio.py path/to/suspect_audio.wav
```
*(Supports `.wav`, `.mp3`, `.flac`, `.m4a`, `.mp4`, `.mkv` files. Outputs formatted JSON to stdout and writes to `output/audio_results/<clip_id>.json`.)*

### 4. Run F1 Standalone Script
```powershell
uv run python src/p_audio/zero_day_detection.py --input path/to/audio.wav --out output/f1_result.json
```

### 5. Run F2 Standalone Script
```powershell
uv run python src/p_audio/replay_detection.py --input path/to/audio.wav --out output/f2_result.json
```

### 6. Run Test Suite
```powershell
uv run pytest tests/test_p_audio.py
```

---

## Machine-Readable Output JSON Contract

```json
{
  "clip_id": "demo_audio_01",
  "status": "SUCCESS",
  "f1_zero_day": {
    "known_generator_similarity": "LOW",
    "natural_speech_consistency": "HIGH",
    "verdict": "REAL",
    "confidence": 0.8166
  },
  "f2_replay": {
    "pathway": "DIRECT_GENUINE",
    "reverb_detected": false,
    "confidence": 0.8679
  },
  "known_generator_similarity": "LOW",
  "natural_speech_consistency": "HIGH",
  "pathway": "DIRECT_GENUINE",
  "reverb_detected": false,
  "verdict": "REAL",
  "confidence": 0.8166,
  "evidence": {
    "preprocessing": {
      "total_duration_sec": 3.5,
      "speech_duration_sec": 3.5,
      "sampling_rate": 16000
    },
    "f1_details": {
      "similarity_score_raw": 0.1706,
      "consistency_score_raw": 0.8166,
      "oc_svm_raw_score": 0.7466
    },
    "f2_evidence": {
      "t60_estimated_sec": 0.05,
      "t60_threshold_sec": 0.3,
      "noise_floor_db": -60.0,
      "snr_db": 30.0,
      "comb_filter_index": 0.04
    },
    "processing_time_sec": 0.184
  }
}
```
