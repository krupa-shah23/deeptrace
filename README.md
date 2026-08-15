# DeepTrace

Forensic authentication system for face-swap deepfake videos — built for an SIH-style problem statement. Given a suspect video, DeepTrace produces a detailed forensic report: whether it's confirmed manipulated, what abnormalities were observed, and the likely underlying technique/pathway of creation (AI synthesis, conventional editing, or re-recorded/replayed).

This is a hackathon prototype (2-day build) using pre-trained models and classical CV/DSP glued into a rules-based fusion engine — not a trained-from-scratch detector.

---

## Architecture

```
                          Suspect Video
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                     ▼
        Video Track                            Audio Track
              │                                     │
   ┌──────────┴──────────┐              ┌───────────┴───────────┐
   ▼                      ▼              ▼                       ▼
Eye-Reflection      Source/Propagation  Zero-Day/Unknown    Replay/Physical-
Consistency (F4)    Forensics (F3)      Generator Det (F1)  World Re-Rec (F2)
   │                      │              │                       │
   └──────────┬───────────┘              └───────────┬───────────┘
              ▼                                       ▼
        video_result.json                       audio_result.json
              └──────────────────┬──────────────────┘
                                  ▼
                    Cause-of-Manipulation Fusion Engine (F5)
                                  │
                                  ▼
                   Forensic Report (PDF/HTML) + Cause Tree
```

## Team & Ownership

| Role | Owns | Folder |
|---|---|---|
| **P-Video** | F4 Eye-Reflection Consistency, F3 Source/Propagation Forensics | `src/p_video/` |
| **P-Audio** | F1 Zero-Day/Unknown-Generator Detection, F2 Replay/Re-Recording Detection | `src/p_audio/` |
| **P-Fusion** | F5 Cause-of-Manipulation Attribution, report generation, demo integration | `src/p_fusion/` |

Each role owns its own folder and writes to a shared `output/` directory using the JSON contracts defined in [`docs/contracts.md`](docs/contracts.md). No one imports code directly from another role's folder — integration happens strictly through JSON files, so all 3 people can work in parallel with zero merge conflicts.

---

## Project Structure

```
deeptrace/
├── pyproject.toml          # single shared dependency file — do not create per-role venvs
├── uv.lock                 # locked exact versions, committed to git
├── README.md
├── docs/
│   └── contracts.md        # JSON I/O contracts between all 3 roles
├── data/
│   ├── seed_clips/         # original demo videos (real + face-swapped)
│   ├── ff++/                # FaceForensics++ subset (gitignored, download separately)
│   └── generated_variants/ # ffmpeg-degraded copies of seed clips, for F3's DB
├── src/
│   ├── shared/
│   │   └── preprocessing.py   # frame extraction — imported by p_video
│   ├── p_video/
│   │   ├── eye_reflection.py     # F4
│   │   └── source_propagation.py # F3
│   ├── p_audio/
│   │   ├── zero_day_detection.py # F1
│   │   └── replay_detection.py   # F2
│   └── p_fusion/
│       ├── fusion_engine.py      # F5 rules engine
│       └── report_generator.py   # PDF/HTML report + cause tree
├── output/
│   ├── video_results/      # F4/F3 JSON, one file per processed clip
│   ├── audio_results/      # F1/F2 JSON
│   └── reports/            # final generated reports
└── tests/
    └── test_demo_set.py    # runs the shared demo clips end-to-end
```

---

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) — no separate Python install needed, `uv` manages it.

```powershell
git clone <repo-url>
cd deeptrace
uv sync
```

`uv sync` installs the **exact locked versions** from `uv.lock` for everyone — don't run `uv add` casually on your own machine without pushing the updated lockfile, or teammates will drift out of sync.

### ffmpeg (required for F3, and for generating demo variants)

Windows:
```powershell
winget install Gyan.FFmpeg
```
Restart your terminal after installing, then confirm:
```powershell
ffmpeg -version
```

Linux:
```bash
sudo apt update && sudo apt install ffmpeg
```

### Verify your setup

```powershell
uv run python -c "import mediapipe as mp; import cv2; import librosa; print('all core imports OK')"
```

---

## Running

Every script runs through `uv run` — never bare `python`:

```powershell
uv run python src/p_video/eye_reflection.py --input data/seed_clips/demo1.mp4
uv run python src/p_audio/zero_day_detection.py --input data/seed_clips/demo1_audio.wav
uv run python src/p_fusion/fusion_engine.py --video-json output/video_results/demo1.json --audio-json output/audio_results/demo1.json
```

To run the full pipeline on the shared demo set:
```powershell
uv run python tests/test_demo_set.py
```

---

## Dependencies (by role)

Everything lives in one `pyproject.toml` so `uv` resolves compatible versions once for the whole team.

```toml
[project]
name = "deeptrace"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # --- P-Video (F4 eye-reflection, F3 source/propagation) ---
    "opencv-python>=4.9",
    "mediapipe>=0.10",
    "imagehash>=4.3",
    "pillow>=10.0",
    "ffmpeg-python>=0.2",

    # --- P-Audio (F1 zero-day detection, F2 replay detection) ---
    "librosa>=0.10",
    "soundfile>=0.12",
    "parselmouth-praat>=0.4",
    "scikit-learn>=1.4",
    "pyroomacoustics>=0.7",
    "torch>=2.2",
    "transformers>=4.38",

    # --- P-Fusion (F5 fusion engine + reporting) ---
    "reportlab>=4.1",
    "jinja2>=3.1",
    "weasyprint>=61.0",
    "cryptography>=42.0",

    # --- shared ---
    "numpy>=1.26",
    "pandas>=2.2",
    "matplotlib>=3.8",
]
```

> **Note on `torch`:** the default install pulls CUDA wheels (~2-3GB). If your machine has no NVIDIA GPU, install the CPU-only build instead to keep `uv sync` fast:
> ```powershell
> uv add torch --index-url https://download.pytorch.org/whl/cpu
> ```
> then commit the updated `uv.lock` so teammates on CPU-only machines get the same lighter install.

---

## Datasets

| Dataset | Used for | Access |
|---|---|---|
| [FaceForensics++](https://github.com/ondyari/FaceForensics) (`c23` split, DeepFakes + FaceSwap subsets) | F4 primary | Request-access form on the repo |
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | F4 secondary/fallback | Request-access form |
| Self-recorded clip + free face-swap tool (`roop`/`insightface`) | F4 guaranteed fallback + live demo | Generate locally |
| Self-generated ffmpeg variants of seed clips | F3 — this *is* the database | Generate locally, see `data/generated_variants/` |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | F1 real-speech baseline | Public download |
| [ASVspoof 2019/2021 LA](https://www.asvspoof.org/) | F1 known-fake baseline | Public download |
| Self-recorded replay pairs (speaker + mic re-recording) | F2 — this *is* the dataset | Generate locally |

None of these are committed to the repo (too large) — `data/` outside of `seed_clips/` and `generated_variants/` is gitignored. Download instructions per dataset live in each role's script docstring.

---

## Integration Contract

Full JSON schemas are in [`docs/contracts.md`](docs/contracts.md). Summary:

- **P-Video → P-Fusion**: `video_results/<clip_id>.json` — eye-reflection verdict + source-match verdict
- **P-Audio → P-Fusion**: `audio_results/<clip_id>.json` — zero-day verdict + replay-pathway verdict
- **P-Fusion** reads both, applies the fusion rules tree, writes `reports/<clip_id>_report.pdf`

Field names in these files are locked by team agreement — don't rename a field without flagging it in the team channel, since P-Fusion's rules engine matches on exact keys.


## Scope Notes

Deliberately out of scope for this prototype: internet-scale source tracing, model training beyond a small one-class SVM (F1), C2PA/blockchain provenance, on-device tokenization, pre-forward interception, real legal-database lookups (a few example BSA Section 63 references are hardcoded in the report template instead).
