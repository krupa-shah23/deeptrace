import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PAudioConfig:
    """Central configuration for P-Audio forensic module."""

    # Preprocessing settings
    target_sr: int = 16000
    min_duration_sec: float = 2.0
    vad_top_db: int = 30  # Threshold below peak considered silence for VAD

    # F1 Model paths & thresholds (Zero-Day / Unknown Generator)
    f1_model_path: Path = field(
        default_factory=lambda: Path("models/f1_one_class_svm.joblib")
    )
    # Dedicated anti-spoofing checkpoint (trained on ASVspoof / synthetic audio detection)
    pretrained_model_name: str = "mohammedgaber/wav2vec2-large-xlsr-53-anti-spoofing"
    f1_similarity_threshold_high: float = 0.70
    f1_similarity_threshold_med: float = 0.40
    f1_consistency_threshold_high: float = 0.0  # OC-SVM score >= 0
    f1_consistency_threshold_med: float = -0.30

    # F2 Thresholds (Replay / Physical-World Re-recording)
    f2_reverb_t60_threshold_sec: float = 0.30  # Estimated T60 > 0.3s indicates reverberant space
    f2_noise_floor_threshold_db: float = -45.0
    f2_comb_filter_threshold: float = 0.25
    f2_spectral_flatness_threshold: float = 0.12

    # Output directory
    output_dir: Path = field(
        default_factory=lambda: Path("output/audio_results")
    )

    def __post_init__(self):
        if isinstance(self.f1_model_path, str):
            self.f1_model_path = Path(self.f1_model_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
