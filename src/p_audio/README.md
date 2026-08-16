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
1. **Pretrained Anti-Spoofing Model**: Computes similarity score against known synthetic vocoder / TTS generation patterns using a fine-tuned neural model or robust acoustic feature embedding baseline.
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
- **Comb Filtering Index**: Autocorrelation of magnitude spectrum residual peaks to detect periodic room reflections.
- **Spectral Flatness & Rolloff**: High-frequency acoustic attenuation.

**Pathways produced by F2:**
- `DIRECT_GENUINE`: Clean direct digital/mic recording of real voice.
- `DIRECT_SYNTHETIC`: Direct digital output of AI voice generator without room playback.
- `REPLAYED_RECORDING`: Audio played through speaker in a physical room and re-recorded.
- `INCONCLUSIVE`: Audio clip unusable or insufficient evidence.

---

## Benchmark Methodology & Real Dataset Setup

> [!IMPORTANT]
> **The benchmark does not use generated placeholder audio.**  
> Benchmark evaluation requires official public datasets (VCTK for real speech baseline, ASVspoof 2019 LA for synthetic/spoofed speech, and ASVspoof 2019 PA for physical replay). If real datasets are missing, benchmark mode **fails loudly** rather than fabricating fake numbers.

### 1. Public Dataset Sources & Provenance

- **VCTK Corpus (0.92)**: Genuine human speech baseline for One-Class SVM training and evaluation.
- **ASVspoof 2019 LA**: Labeled synthetic/spoofed speech for F1 known-attack (`A01-A06`) and zero-day held-out attack (`A07-A19`) evaluation.
- **ASVspoof 2019 PA**: Physical-world re-recording and acoustic replay evaluation for F2.

### 2. Dataset Preparation & Manifest Setup

To setup and parse public datasets:
```powershell
# 1. Download official dataset release archives
uv run python scripts/prepare_datasets.py --download

# 2. Build machine-readable CSV manifests & metadata
uv run python scripts/build_manifests.py

# 3. Verify dataset health and speaker-disjoint constraint
uv run python scripts/verify_datasets.py
```

### 3. Speaker-Disjoint Policy & Zero-Leakage Guarantee
- **Train Split**: Genuine real human speech from `Train` speakers ONLY.
- **Val Split**: Genuine + Spoof speech from `Val` speakers (used strictly for threshold calibration).
- **Test Split**: Genuine + Spoof speech from `Test` speakers (completely speaker-disjoint; 0 speaker overlap with Train).

### 4. Zero-Day Evaluation Strategy
ASVspoof 2019 LA contains distinct attack types:
- **Known Attacks (`A01-A06`)**: Present in training/dev protocols.
- **Zero-Day Held-Out Attacks (`A07-A19`)**: Present ONLY in evaluation test protocol.
The evaluation script reports accuracy separately for known vs zero-day attacks.

### 5. Running Full End-to-End Benchmark
```powershell
uv run python scripts/run_benchmark.py --mode benchmark
```
Outputs machine-readable metrics (`output/benchmark/f1_metrics.json`, `f2_metrics.json`, `per_attack_results.csv`) and human-readable Markdown report (`output/benchmark/benchmark_report.md`).

---

## Demo & Developer Smoke Testing

For single-file inference or quick developer tests without downloading full benchmark datasets:

### 1. Run Complete Pipeline on Single Clip
```powershell
uv run python scripts/run_audio.py path/to/suspect_audio.wav
```

### 2. Run Smoke-Test Benchmark
```powershell
uv run python scripts/run_benchmark.py --mode smoke-test
```

### 3. Run Unit Test Suite
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
      "oc_svm_raw_score": 0.7466,
      "pretrained_model": {
        "model_name": "mohammedgaber/wav2vec2-large-xlsr-53-anti-spoofing",
        "mode": "fine_tuned_anti_spoof_neural"
      }
    },
    "f2_evidence": {
      "t60_estimated_sec": 0.05,
      "t60_threshold_sec": 0.3,
      "noise_floor_db": -60.0,
      "snr_db": 30.0,
      "comb_filter_index": 0.04,
      "threshold_status": "CALIBRATED"
    },
    "processing_time_sec": 0.184
  }
}
```
