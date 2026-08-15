from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import librosa
from scipy import signal, stats


@dataclass
class AudioFeatures:
    """Dataclass holding extracted feature vector and detailed metrics dictionary."""
    feature_vector: np.ndarray
    feature_names: list
    details: Dict[str, Any]


def safe_nan_clean(val: float, default: float = 0.0) -> float:
    """Helper to clean NaN or Inf values into safe floats."""
    if np.isnan(val) or np.isinf(val):
        return default
    return float(val)


def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Tuple[np.ndarray, list, Dict[str, float]]:
    """Extracts MFCCs and delta statistics."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    delta_means = np.mean(mfcc_delta, axis=1)
    delta_stds = np.std(mfcc_delta, axis=1)

    vector = np.concatenate([means, stds, delta_means, delta_stds])
    names = [f"mfcc_{i}_mean" for i in range(n_mfcc)] + \
            [f"mfcc_{i}_std" for i in range(n_mfcc)] + \
            [f"mfcc_delta_{i}_mean" for i in range(n_mfcc)] + \
            [f"mfcc_delta_{i}_std" for i in range(n_mfcc)]

    details = {
        "mfcc_mean_1": safe_nan_clean(means[1] if len(means) > 1 else 0.0),
        "mfcc_std_1": safe_nan_clean(stds[1] if len(stds) > 1 else 0.0)
    }
    return vector, names, details


def extract_lfcc_features(y: np.ndarray, sr: int, n_lfcc: int = 13) -> Tuple[np.ndarray, list, Dict[str, float]]:
    """Extracts Linear Frequency Cepstral Coefficients (LFCC) statistics."""
    # Compute linear filterbank STFT
    S = np.abs(librosa.stft(y))
    # Linear spaced filterbank approximation using scipy signal
    n_fft = (S.shape[0] - 1) * 2
    linear_fbuf = np.linspace(0, sr / 2.0, S.shape[0])
    # Compute log spectral energy across linear bands
    log_S = np.log(S + 1e-8)
    # DCT type II for LFCC
    lfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=n_lfcc, dct_type=2)

    means = np.mean(lfcc, axis=1)
    stds = np.std(lfcc, axis=1)

    vector = np.concatenate([means, stds])
    names = [f"lfcc_{i}_mean" for i in range(n_lfcc)] + [f"lfcc_{i}_std" for i in range(n_lfcc)]
    details = {
        "lfcc_mean_0": safe_nan_clean(means[0] if len(means) > 0 else 0.0)
    }
    return vector, names, details


def extract_pitch_microtremor_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, list, Dict[str, float]]:
    """
    Extracts fundamental frequency (F0) statistics, pitch micro-tremor,
    Jitter (local period perturbation quotient), and Shimmer (amplitude perturbation quotient).
    """
    # Estimate F0 pitch track using PYIN pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C6'),
        sr=sr
    )

    valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]

    if len(valid_f0) > 3:
        f0_mean = float(np.mean(valid_f0))
        f0_std = float(np.std(valid_f0))
        f0_min = float(np.min(valid_f0))
        f0_max = float(np.max(valid_f0))

        # Pitch periods (in seconds)
        periods = 1.0 / valid_f0
        # Jitter: relative average difference between consecutive pitch periods
        period_diffs = np.abs(np.diff(periods))
        jitter_local = float(np.mean(period_diffs) / np.mean(periods)) if np.mean(periods) > 1e-6 else 0.0

        # Pitch variation coefficient
        f0_cv = float(f0_std / f0_mean) if f0_mean > 1e-6 else 0.0
    else:
        f0_mean, f0_std, f0_min, f0_max = 0.0, 0.0, 0.0, 0.0
        jitter_local = 0.0
        f0_cv = 0.0

    # Shimmer calculation using amplitude peak difference across frame boundaries
    frame_length = 512
    hop_length = 256
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    amplitudes = np.max(np.abs(frames), axis=0)
    valid_amps = amplitudes[amplitudes > 1e-4]

    if len(valid_amps) > 3:
        amp_diffs = np.abs(np.diff(valid_amps))
        shimmer_local = float(np.mean(amp_diffs) / np.mean(valid_amps)) if np.mean(valid_amps) > 1e-6 else 0.0
    else:
        shimmer_local = 0.0

    vector = np.array([
        safe_nan_clean(f0_mean),
        safe_nan_clean(f0_std),
        safe_nan_clean(f0_cv),
        safe_nan_clean(jitter_local),
        safe_nan_clean(shimmer_local)
    ])
    names = ["f0_mean", "f0_std", "f0_cv", "jitter_local", "shimmer_local"]

    details = {
        "f0_mean_hz": safe_nan_clean(f0_mean),
        "f0_std_hz": safe_nan_clean(f0_std),
        "jitter_local": safe_nan_clean(jitter_local),
        "shimmer_local": safe_nan_clean(shimmer_local)
    }
    return vector, names, details


def extract_phase_features(y: np.ndarray) -> Tuple[np.ndarray, list, Dict[str, float]]:
    """Extracts STFT phase continuity and phase spectrum statistics."""
    stft = librosa.stft(y)
    phase = np.angle(stft)

    # Phase derivative across time (instantaneous frequency deviation proxy)
    phase_diff_time = np.diff(phase, axis=1)
    # Phase derivative across frequency
    phase_diff_freq = np.diff(phase, axis=0)

    phase_var_time = float(np.var(phase_diff_time)) if phase_diff_time.size > 0 else 0.0
    phase_var_freq = float(np.var(phase_diff_freq)) if phase_diff_freq.size > 0 else 0.0
    phase_mean = float(np.mean(phase)) if phase.size > 0 else 0.0

    vector = np.array([
        safe_nan_clean(phase_mean),
        safe_nan_clean(phase_var_time),
        safe_nan_clean(phase_var_freq)
    ])
    names = ["phase_mean", "phase_var_time", "phase_var_freq"]

    details = {
        "phase_var_time": safe_nan_clean(phase_var_time),
        "phase_var_freq": safe_nan_clean(phase_var_freq)
    }
    return vector, names, details


def extract_acoustic_replay_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, list, Dict[str, float]]:
    """
    Extracts acoustic evidence for F2 (Replay / Re-recording Detection):
    - Reverberation / T60 decay proxy (Schroeder energy decay curve slope)
    - Noise floor level (dB)
    - Signal-to-Noise Ratio (SNR dB)
    - Spectral Flatness & Spectral Rolloff
    - Comb filtering index (spectral autocorrelation peak ratio)
    """
    # 1. Reverberation / T60 proxy via Energy Decay Curve (EDC)
    # Compute short-term energy
    hop = 160
    frame_len = 320
    energy = np.array([
        np.sum(y[i:i + frame_len] ** 2)
        for i in range(0, len(y) - frame_len, hop)
    ])
    if len(energy) > 10:
        # Backward integration (Schroeder integration)
        schroeder_edc = np.cumsum(energy[::-1])[::-1]
        schroeder_edc_db = 10 * np.log10(schroeder_edc / (np.max(schroeder_edc) + 1e-10) + 1e-10)

        # Estimate T60 from slope between -5 dB and -25 dB (EDT / T20 extension)
        valid_idx = np.where((schroeder_edc_db <= -5) & (schroeder_edc_db >= -25))[0]
        if len(valid_idx) > 2:
            times = valid_idx * (hop / float(sr))
            slope, _, _, _, _ = stats.linregress(times, schroeder_edc_db[valid_idx])
            t60_est = float(-60.0 / slope) if slope < -1e-5 else 0.05
        else:
            t60_est = 0.05
    else:
        t60_est = 0.05

    # Clamp T60 proxy to realistic physical bounds [0.0, 3.0s]
    t60_est = float(np.clip(t60_est, 0.0, 3.0))

    # 2. Noise floor (dB) and SNR (dB)
    # Estimate noise floor from bottom 10th percentile frame energy
    if len(energy) > 5:
        sorted_energy = np.sort(energy)
        noise_energy = np.mean(sorted_energy[:max(1, int(len(sorted_energy) * 0.10))])
        signal_energy = np.mean(sorted_energy[int(len(sorted_energy) * 0.50):])

        noise_floor_db = float(10 * np.log10(noise_energy + 1e-10))
        snr_db = float(10 * np.log10((signal_energy + 1e-10) / (noise_energy + 1e-10)))
    else:
        noise_floor_db = -60.0
        snr_db = 30.0

    # 3. Spectral Flatness & Rolloff
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness)) if flatness.size > 0 else 0.0

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    mean_rolloff = float(np.mean(rolloff)) if rolloff.size > 0 else 0.0

    # 4. Comb Filtering Index
    # Re-recording through speaker-room-mic path leaves periodic spectral notch ripples
    S = np.abs(librosa.stft(y))
    avg_spec = np.mean(S, axis=1)
    if len(avg_spec) > 20:
        autocorr_spec = np.correlate(avg_spec - np.mean(avg_spec), avg_spec - np.mean(avg_spec), mode='full')
        autocorr_spec = autocorr_spec[len(autocorr_spec) // 2:]
        if autocorr_spec[0] > 1e-6:
            autocorr_spec /= autocorr_spec[0]
            # Secondary peak in spectral autocorrelation indicates comb filtering
            comb_filter_index = float(np.max(autocorr_spec[3:50])) if len(autocorr_spec) > 50 else 0.0
        else:
            comb_filter_index = 0.0
    else:
        comb_filter_index = 0.0

    vector = np.array([
        safe_nan_clean(t60_est),
        safe_nan_clean(noise_floor_db),
        safe_nan_clean(snr_db),
        safe_nan_clean(mean_flatness),
        safe_nan_clean(mean_rolloff),
        safe_nan_clean(comb_filter_index)
    ])
    names = ["t60_est_sec", "noise_floor_db", "snr_db", "spectral_flatness", "spectral_rolloff_hz", "comb_filter_index"]

    details = {
        "t60_est_sec": safe_nan_clean(t60_est),
        "noise_floor_db": safe_nan_clean(noise_floor_db),
        "snr_db": safe_nan_clean(snr_db),
        "spectral_flatness": safe_nan_clean(mean_flatness),
        "spectral_rolloff_hz": safe_nan_clean(mean_rolloff),
        "comb_filter_index": safe_nan_clean(comb_filter_index)
    }
    return vector, names, details


def extract_all_audio_features(y: np.ndarray, sr: int) -> AudioFeatures:
    """
    Master feature extraction pipeline extracting MFCCs, LFCCs, Pitch/Microtremor,
    STFT Phase continuity, and Acoustic Replay metrics into a unified, clean numerical vector.
    """
    mfcc_v, mfcc_n, mfcc_d = extract_mfcc_features(y, sr)
    lfcc_v, lfcc_n, lfcc_d = extract_lfcc_features(y, sr)
    pitch_v, pitch_n, pitch_d = extract_pitch_microtremor_features(y, sr)
    phase_v, phase_n, phase_d = extract_phase_features(y)
    replay_v, replay_n, replay_d = extract_acoustic_replay_features(y, sr)

    feature_vector = np.concatenate([mfcc_v, lfcc_v, pitch_v, phase_v, replay_v])
    feature_names = mfcc_n + lfcc_n + pitch_n + phase_n + replay_n

    # Replace any residual NaNs or Infs
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    combined_details = {}
    combined_details.update(mfcc_d)
    combined_details.update(lfcc_d)
    combined_details.update(pitch_d)
    combined_details.update(phase_d)
    combined_details.update(replay_d)

    return AudioFeatures(
        feature_vector=feature_vector,
        feature_names=feature_names,
        details=combined_details
    )
