"""Signal processing helpers for ECG analysis."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

DEFAULT_SAMPLING_RATE = 360.0  # Hz, common MIT-BIH sampling rate

R_PEAK_PROMINENCE_SCALE = 0.6  # scales the std-dev to decide peak prominence
R_PEAK_DISTANCE_SECONDS = 0.2  # minimum spacing between beats
HEARTBEAT_PRE_SECONDS = 0.25
HEARTBEAT_POST_SECONDS = 0.45
HEARTBEAT_SPECTRAL_BINS = 128
MAX_BEATS_FOR_PLOT = 30


@dataclass
class FFTResult:
    frequencies: np.ndarray
    magnitudes: np.ndarray
    spectrum: np.ndarray


def load_signal_from_csv(file_bytes: bytes) -> np.ndarray:
    """Return ECG signal vector from CSV bytes."""
    try:
        decoded = file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as err:
        raise ValueError("Unable to decode file as UTF-8 text.") from err

    data: list[float] = []
    first_data_line = True
    for line in decoded.splitlines():
        line = line.strip()
        if not line:
            continue
        value = line.split(",")[0]
        try:
            data.append(float(value))
            first_data_line = False
        except ValueError:
            # If the *first* non-empty line is non-numeric, treat it as a header and skip it.
            if first_data_line:
                first_data_line = False
                continue
            raise ValueError(f"Invalid numeric value '{value}' in CSV (after header).")

    arr = np.asarray(data, dtype=np.float64)
    if arr.size < 8:
        raise ValueError("Signal must contain at least 8 samples.")
    return arr


def compute_fft(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> FFTResult:
    """Compute single-sided FFT magnitude spectrum."""
    # Remove DC offset so the 0 Hz bin does not dominate the spectrum.
    centered = signal.astype(np.float64) - float(np.mean(signal))

    n = centered.size
    spectrum = np.fft.fft(centered)
    freqs = np.fft.fftfreq(n, d=1.0 / sampling_rate)

    half = n // 2
    magnitudes = 2.0 / n * np.abs(spectrum[:half])
    return FFTResult(frequencies=freqs[:half], magnitudes=magnitudes, spectrum=spectrum)


def detect_r_peaks(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> np.ndarray:
    """Detect candidate R-peaks using SciPy's prominence-based peak finder."""
    if signal.size < 4:
        return np.array([], dtype=int)

    distance = max(int(R_PEAK_DISTANCE_SECONDS * sampling_rate), 1)
    prominence = max(np.std(signal) * R_PEAK_PROMINENCE_SCALE, 0.1)
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    if peaks.size == 0:
        return np.array([int(np.argmax(signal))], dtype=int)
    return peaks.astype(int)


def extract_aligned_beats(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Center beats around each R-peak to create phase-synchronized segments."""
    pre = max(int(HEARTBEAT_PRE_SECONDS * sampling_rate), 1)
    post = max(int(HEARTBEAT_POST_SECONDS * sampling_rate), 1)
    window = pre + post

    peaks = detect_r_peaks(signal, sampling_rate)
    beats: List[np.ndarray] = []
    kept_peaks: List[int] = []

    for peak in peaks:
        start = peak - pre
        end = peak + post
        if start < 0 or end >= signal.size:
            continue
        beat = signal[start:end].astype(np.float64)
        beat -= beat.mean()
        beats.append(beat)
        kept_peaks.append(peak)

    if not beats:
        fallback = signal[:window]
        if fallback.size < window:
            fallback = np.pad(fallback, (0, window - fallback.size))
        beats = [fallback - np.mean(fallback)]
        kept_peaks = [int(np.argmax(signal))]

    return np.vstack(beats), np.asarray(kept_peaks, dtype=int)


def build_feature_matrix(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spectral feature matrix from heartbeat-aligned FFT magnitudes."""
    beats, r_peaks = extract_aligned_beats(signal, sampling_rate)
    window = np.hamming(beats.shape[1])
    windowed = beats * window
    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    max_bins = min(HEARTBEAT_SPECTRAL_BINS, spectra.shape[1])
    spectra = spectra[:, :max_bins]
    norms = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-12
    features = spectra / norms
    return features, beats, r_peaks


def run_pca(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run PCA and return transformed coordinates plus explained variance."""
    rows = feature_matrix.shape[0]
    if rows < 2:
        feature_matrix = np.vstack([feature_matrix, feature_matrix])

    pca = PCA(n_components=2, whiten=True, random_state=42)
    components = pca.fit_transform(feature_matrix)
    return components, pca.explained_variance_ratio_


def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode Matplotlib figure as base64 data URI."""
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    import base64

    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_time_plot(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> str:
    """Return base64 image for time-domain ECG plot."""
    times = np.arange(signal.size) / sampling_rate
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, signal, color="#2563eb", linewidth=1)
    ax.set_title("ECG Signal (Time Domain)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def create_frequency_plot(fft_res: FFTResult) -> str:
    """Return base64 image for FFT magnitude plot."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(fft_res.frequencies, fft_res.magnitudes, color="#16a34a", linewidth=1)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    # Focus on the clinically relevant ECG band; zoom into low frequencies
    max_freq = float(fft_res.frequencies.max())
    ax.set_xlim(0, min(50.0, max_freq))
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def create_aligned_heartbeat_plot(beats: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> str:
    """Return base64 image showing aligned heartbeats plus their template."""
    if beats.size == 0:
        raise ValueError("No beats available for visualization.")

    pre = max(int(HEARTBEAT_PRE_SECONDS * sampling_rate), 1)
    total_samples = beats.shape[1]
    times = (np.arange(total_samples) - pre) / sampling_rate
    fig, ax = plt.subplots(figsize=(6, 3))

    subset = beats[: min(beats.shape[0], MAX_BEATS_FOR_PLOT)]
    ax.plot(times, subset.T, color="#38bdf8", alpha=0.2)
    ax.plot(times, subset.mean(axis=0), color="#f97316", linewidth=2, label="Template")
    ax.axvline(0, color="#facc15", linestyle="--", linewidth=1, label="R-peak")
    ax.set_title("Aligned Heartbeats")
    ax.set_xlabel("Time relative to R-peak (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    return _fig_to_base64(fig)


def create_pca_plot(components: np.ndarray) -> str:
    """Return base64 image visualizing top 2 PCA components."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(components[:, 0], components[:, 1], c=np.linspace(0, 1, components.shape[0]), cmap="viridis")
    ax.set_title("PCA on Spectral Features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


