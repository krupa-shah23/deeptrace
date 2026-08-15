import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from src.p_audio.config import PAudioConfig


@dataclass
class PreprocessedAudio:
    """Dataclass holding preprocessed audio signal and metadata."""
    file_path: str
    audio: Optional[np.ndarray]
    sr: int
    duration_sec: float
    speech_duration_sec: float
    usable: bool
    reason: str


def extract_audio_from_video(video_path: str, target_sr: int = 16000) -> Optional[str]:
    """
    Extracts audio track from a video container (MP4, MKV, AVI, etc.) to a temporary WAV file using ffmpeg.
    Returns path to temporary WAV file or None if extraction failed.
    """
    try:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",  # no video
            "-ac", "1",  # mono
            "-ar", str(target_sr),  # sample rate
            "-f", "wav",
            temp_wav_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            return temp_wav_path
    except Exception as e:
        print(f"[preprocess] ffmpeg video audio extraction failed for {video_path}: {e}")
    return None


def load_and_preprocess_audio(
    file_path: str,
    config: Optional[PAudioConfig] = None
) -> PreprocessedAudio:
    """
    Loads audio file (or extracts from video), converts to mono, resamples to target_sr,
    performs VAD/silence trimming, and normalizes amplitude safely.
    """
    if config is None:
        config = PAudioConfig()

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        return PreprocessedAudio(
            file_path=file_path,
            audio=None,
            sr=config.target_sr,
            duration_sec=0.0,
            speech_duration_sec=0.0,
            usable=False,
            reason=f"File not found: {file_path}"
        )

    temp_extracted_wav = None
    actual_path = file_path

    # Check if input is a video file or non-standard format
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
    if file_path_obj.suffix.lower() in video_extensions:
        extracted = extract_audio_from_video(file_path, config.target_sr)
        if extracted is not None:
            actual_path = extracted
            temp_extracted_wav = extracted
        else:
            return PreprocessedAudio(
                file_path=file_path,
                audio=None,
                sr=config.target_sr,
                duration_sec=0.0,
                speech_duration_sec=0.0,
                usable=False,
                reason="Failed to extract audio track from video file"
            )

    try:
        # Load audio with librosa
        audio, sr = librosa.load(actual_path, sr=config.target_sr, mono=True)
    except Exception as e:
        # Retry with ffmpeg extraction if librosa failed
        if temp_extracted_wav is None:
            extracted = extract_audio_from_video(file_path, config.target_sr)
            if extracted is not None:
                temp_extracted_wav = extracted
                try:
                    audio, sr = librosa.load(extracted, sr=config.target_sr, mono=True)
                except Exception as inner_e:
                    return PreprocessedAudio(
                        file_path=file_path,
                        audio=None,
                        sr=config.target_sr,
                        duration_sec=0.0,
                        speech_duration_sec=0.0,
                        usable=False,
                        reason=f"Audio loading failed: {inner_e}"
                    )
            else:
                return PreprocessedAudio(
                    file_path=file_path,
                    audio=None,
                    sr=config.target_sr,
                    duration_sec=0.0,
                    speech_duration_sec=0.0,
                    usable=False,
                    reason=f"Audio loading failed: {e}"
                )
        else:
            return PreprocessedAudio(
                file_path=file_path,
                audio=None,
                sr=config.target_sr,
                duration_sec=0.0,
                speech_duration_sec=0.0,
                usable=False,
                reason=f"Audio loading failed: {e}"
            )
    finally:
        # Clean up temporary WAV if created
        if temp_extracted_wav and os.path.exists(temp_extracted_wav):
            try:
                os.remove(temp_extracted_wav)
            except Exception:
                pass

    if audio is None or len(audio) == 0:
        return PreprocessedAudio(
            file_path=file_path,
            audio=None,
            sr=config.target_sr,
            duration_sec=0.0,
            speech_duration_sec=0.0,
            usable=False,
            reason="Audio file is empty or corrupted"
        )

    # Calculate initial total duration
    total_duration_sec = len(audio) / float(sr)

    # Silence check for pure silence or near-zero amplitude audio
    if np.max(np.abs(audio)) < 1e-4:
        return PreprocessedAudio(
            file_path=file_path,
            audio=audio,
            sr=sr,
            duration_sec=total_duration_sec,
            speech_duration_sec=0.0,
            usable=False,
            reason="No active speech detected (silence only)"
        )

    # Perform VAD / silence trimming using energy top_db
    intervals = librosa.effects.split(audio, top_db=config.vad_top_db)
    if len(intervals) == 0:
        return PreprocessedAudio(
            file_path=file_path,
            audio=audio,
            sr=sr,
            duration_sec=total_duration_sec,
            speech_duration_sec=0.0,
            usable=False,
            reason="No active speech detected (silence only)"
        )

    # Concatenate active speech frames
    speech_frames = np.concatenate([audio[start:end] for start, end in intervals])
    speech_duration_sec = len(speech_frames) / float(sr)

    # Check minimum speech duration requirement
    if speech_duration_sec < config.min_duration_sec:
        return PreprocessedAudio(
            file_path=file_path,
            audio=speech_frames,
            sr=sr,
            duration_sec=total_duration_sec,
            speech_duration_sec=speech_duration_sec,
            usable=False,
            reason=f"Audio clip speech duration ({speech_duration_sec:.2f}s) is below minimum required duration ({config.min_duration_sec}s)"
        )

    # Safe Peak / RMS Normalization (prevent division by zero and clipping)
    max_val = np.max(np.abs(speech_frames))
    if max_val > 1e-6:
        normalized_audio = speech_frames / max_val * 0.95
    else:
        normalized_audio = speech_frames

    return PreprocessedAudio(
        file_path=file_path,
        audio=normalized_audio,
        sr=sr,
        duration_sec=total_duration_sec,
        speech_duration_sec=speech_duration_sec,
        usable=True,
        reason="OK"
    )
