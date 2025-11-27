## Backend Code – Line‑by‑Line Style Explanation (Combined)

This document explains the main backend files:

- `main.py` – FastAPI wiring, HTTP endpoints.
- `processing.py` – signal processing and plotting.
- `schemas.py` – Pydantic models (API schemas).
- `math_content.py` – LaTeX formulas for the frontend.
- `run_cli.py` – command‑line interface to print outputs in the terminal.

You can export this Markdown as **one PDF** for code explanation.

---

## 1. `main.py` – FastAPI application

```python
"""FastAPI application wiring ECG analysis helpers to HTTP endpoints."""
```
- **Docstring**: Short description – this file connects the maths / processing code to HTTP endpoints.

```python
from __future__ import annotations
```
- **Future import**: Enables postponed evaluation of annotations (nice for type hints).

```python
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
```
- **Imports from FastAPI**:
  - `FastAPI`: class used to create the web app.
  - `File`, `UploadFile`: helpers to receive uploaded files (CSV) through HTTP.
  - `HTTPException`: used to return error responses.
  - `CORSMiddleware`: middleware to allow cross‑origin requests (frontend on another port).

```python
from .math_content import FORMULA_BLOCKS
from .processing import (
    build_feature_matrix,
    compute_fft,
    create_aligned_heartbeat_plot,
    create_frequency_plot,
    create_pca_plot,
    create_time_plot,
    load_signal_from_csv,
    run_pca,
)
from .schemas import AnalysisResponse
```
- **Local imports**:
  - `FORMULA_BLOCKS`: pre‑defined LaTeX formulas to send to the frontend.
  - From `processing`:
    - `load_signal_from_csv`: converts uploaded CSV bytes into a NumPy array.
    - `compute_fft`: computes FFT of the full ECG.
    - `build_feature_matrix`: builds heartbeat‑level spectral features.
    - `run_pca`: runs PCA on the feature matrix.
    - `create_*_plot`: build base64‑encoded PNGs of various plots.
  - `AnalysisResponse`: Pydantic model describing the JSON structure returned to the frontend.

```python
APP_NAME = "Fourier + PCA ECG Signal Analysis API"
API_PREFIX = "/api"
```
- **Constants**:
  - `APP_NAME`: human‑readable name of the API.
  - `API_PREFIX`: common prefix for all endpoints.

```python
app = FastAPI(title=APP_NAME, version="1.0.0")
```
- **FastAPI instance**: creates the web app object with a title and version.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- **CORS configuration**:
  - Adds middleware that allows requests from any origin (`"*"`).
  - Allows all methods and headers (useful for development and for the React frontend).

```python
def _build_steps(signal_len: int, feature_rows: int, heartbeat_count: int) -> list[dict[str, str]]:
    """Return descriptive pipeline steps for UI timeline."""
```
- **Helper function definition**: builds a list of step dictionaries for the frontend timeline.
  - Parameters:
    - `signal_len`: number of ECG samples.
    - `feature_rows`: number of beats (rows) in the feature matrix.
    - `heartbeat_count`: number of detected R‑peaks / beats.

```python
    return [
        {
            "title": "Load ECG signal",
            "details": f"Parsed {signal_len:,} samples from the uploaded CSV and removed empty rows.",
        },
        {
            "title": "R-peak alignment",
            "details": f"Detected {heartbeat_count} R-peaks and centered each heartbeat to synchronize phase.",
        },
        {
            "title": "Adaptive Fourier projection",
            "details": "Converted every aligned beat to the frequency domain to capture spectral fingerprints.",
        },
        {
            "title": "PCA",
            "details": f"Projected the spectral feature matrix (shape {feature_rows}×F) to 2D embeddings.",
        },
    ]
```
- **Return value**: a list of 4 dictionaries, each with:
  - `title`: step name.
  - `details`: human‑readable description with numbers formatted (like `1,234`).

```python
@app.get(f"{API_PREFIX}/health")
def healthcheck() -> dict[str, str]:
    """Lightweight readiness probe."""
    return {"status": "ok"}
```
- **Health endpoint**:
  - Decorator `@app.get(...)` registers a GET endpoint `/api/health`.
  - Function returns a simple JSON `{"status": "ok"}` to show the server is alive.

```python
@app.post(f"{API_PREFIX}/analyze", response_model=AnalysisResponse)
async def analyze_ecg(file: UploadFile = File(...)) -> AnalysisResponse:
    """Accept ECG CSV, run FFT + PCA, and return plots plus numeric outputs."""
```
- **Main analysis endpoint**:
  - POST endpoint `/api/analyze`.
  - `response_model=AnalysisResponse` ensures the response matches the schema.
  - Parameter `file: UploadFile = File(...)`:
    - Requires a file upload in the request.
    - This is the ECG CSV.

```python
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
```
- **File type validation**:
  - If the uploaded file does not end with `.csv`, raise an HTTP 400 error.

```python
    try:
        raw_bytes = await file.read()
        signal = load_signal_from_csv(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```
- **Reading and parsing CSV**:
  - `file.read()` (awaited) gives the raw bytes.
  - `load_signal_from_csv` converts bytes to a NumPy array of floats.
  - If parsing fails (`ValueError`), translate it into an HTTP 400 response with the error message.

```python
    fft_res = compute_fft(signal)
    feature_matrix, beats, r_peaks = build_feature_matrix(signal)
    pca_components, variance = run_pca(feature_matrix)
```
- **Core computations**:
  - `compute_fft`: FFT on full signal (for global spectrum).
  - `build_feature_matrix`:
    - Detect R‑peaks.
    - Align beats.
    - Build per‑beat spectral features.
  - `run_pca`: PCA on feature matrix – returns:
    - `pca_components`: transformed coordinates.
    - `variance`: explained variance ratios.

```python
    method_summary = (
        "Adaptive Heartbeat-Aligned Fourier PCA centers each detected heartbeat at its R-peak, "
        "applies the Fourier transform per beat to capture spectral envelopes, and runs PCA on the "
        "resulting frequency matrix to obtain low-noise embeddings."
    )
```
- **Text description**:
  - Human‑readable summary of the full method to show in the frontend and CLI.

```python
    plots = [
        {"title": "ECG Signal (Time Domain)", "image_base64": create_time_plot(signal)},
        {"title": "FFT Magnitude Spectrum", "image_base64": create_frequency_plot(fft_res)},
        {"title": "Aligned Heartbeats", "image_base64": create_aligned_heartbeat_plot(beats)},
        {"title": "PCA Scatter Plot", "image_base64": create_pca_plot(pca_components)},
    ]
```
- **Plot data**:
  - Creates a list of plot descriptors, each with:
    - `title`: label for the card in the UI.
    - `image_base64`: base64 PNG from helper plotting functions.

```python
    response = AnalysisResponse(
        fft_frequencies=fft_res.frequencies.tolist(),
        fft_magnitudes=fft_res.magnitudes.tolist(),
        pca_components=pca_components.tolist(),
        explained_variance=variance.tolist(),
        plots=plots,
        formulas=FORMULA_BLOCKS,
        steps=_build_steps(signal_len=signal.size, feature_rows=feature_matrix.shape[0], heartbeat_count=int(len(r_peaks))),
        heartbeat_count=int(len(r_peaks)),
        r_peak_indices=r_peaks.tolist(),
        method_summary=method_summary,
    )
    return response
```
- **Response construction**:
  - Converts NumPy arrays to plain Python lists using `.tolist()`.
  - Includes:
    - FFT frequencies and magnitudes.
    - PCA components and explained variance.
    - Plot metadata and images.
    - Math formulas (`FORMULA_BLOCKS`).
    - Step descriptions (from `_build_steps`).
    - Heartbeat count and R‑peak indices.
    - Method summary string.
  - Returns an `AnalysisResponse` object which FastAPI will serialize to JSON.

```python
__all__ = ["app"]
```
- **Export control**:
  - Indicates this module’s main public object is `app` (used by ASGI servers like Uvicorn).

---

## 2. `processing.py` – Signal Processing and Plots

```python
"""Signal processing helpers for ECG analysis."""
```
- High‑level description: this file contains all core numerical operations.

```python
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple
```
- Future annotations as before.
- `dataclass`: used to define `FFTResult`.
- `BytesIO`: in‑memory byte buffer for images.
- `List`, `Tuple`: type hints for function signatures.

```python
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
```
- Main libraries:
  - `numpy`: numerical arrays and FFT operations.
  - `matplotlib.pyplot`: plotting.
  - `find_peaks`: R‑peak detection.
  - `PCA`: principal component analysis implementation.

```python
DEFAULT_SAMPLING_RATE = 360.0  # Hz, common MIT-BIH sampling rate

R_PEAK_PROMINENCE_SCALE = 0.6  # scales the std-dev to decide peak prominence
R_PEAK_DISTANCE_SECONDS = 0.2  # minimum spacing between beats
HEARTBEAT_PRE_SECONDS = 0.25
HEARTBEAT_POST_SECONDS = 0.45
HEARTBEAT_SPECTRAL_BINS = 128
MAX_BEATS_FOR_PLOT = 30
```
- **Constants**:
  - `DEFAULT_SAMPLING_RATE`: default ECG sampling rate.
  - `R_PEAK_PROMINENCE_SCALE`: how strong peaks must be relative to noise.
  - `R_PEAK_DISTANCE_SECONDS`: minimum distance between peaks in seconds.
  - `HEARTBEAT_PRE_SECONDS`, `HEARTBEAT_POST_SECONDS`: window sizes around each R‑peak.
  - `HEARTBEAT_SPECTRAL_BINS`: number of FFT bins kept as features.
  - `MAX_BEATS_FOR_PLOT`: limit on how many beats to show in the heartbeat alignment plot.

```python
@dataclass
class FFTResult:
    frequencies: np.ndarray
    magnitudes: np.ndarray
    spectrum: np.ndarray
```
- **Dataclass** `FFTResult`:
  - Bundles raw FFT outputs into a structured object:
    - `frequencies`: frequency axis.
    - `magnitudes`: single‑sided magnitude spectrum.
    - `spectrum`: full complex FFT result.

```python
def load_signal_from_csv(file_bytes: bytes) -> np.ndarray:
    """Return ECG signal vector from CSV bytes."""
```
- Function to convert uploaded CSV bytes to a 1D NumPy array.

```python
    try:
        decoded = file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError as err:
        raise ValueError("Unable to decode file as UTF-8 text.") from err
```
- Decodes bytes into UTF‑8 text, strips whitespace; on failure, raises `ValueError`.

```python
    data = []
    for line in decoded.splitlines():
        line = line.strip()
        if not line:
            continue
        value = line.split(",")[0]
        try:
            data.append(float(value))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value '{value}' in CSV.") from exc
```
- Parses each line:
  - Ignores empty lines.
  - Splits on comma and takes the first column.
  - Converts to `float`; if conversion fails, raises a descriptive `ValueError`.

```python
    arr = np.asarray(data, dtype=np.float64)
    if arr.size < 8:
        raise ValueError("Signal must contain at least 8 samples.")
    return arr
```
- Converts list to NumPy array.
- Ensures minimum length (8 samples).
- Returns the array.

```python
def compute_fft(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> FFTResult:
    """Compute single-sided FFT magnitude spectrum."""
```
- Computes FFT and returns `FFTResult`.

```python
    n = signal.size
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1.0 / sampling_rate)
```
- `n`: length of signal.
- `spectrum`: full complex FFT.
- `freqs`: frequency bins corresponding to indices.

```python
    half = n // 2
    magnitudes = 2.0 / n * np.abs(spectrum[:half])
    return FFTResult(frequencies=freqs[:half], magnitudes=magnitudes, spectrum=spectrum)
```
- Uses only first half (positive frequencies) and scales magnitudes.
- Returns frequencies, magnitudes, and original spectrum.

```python
def detect_r_peaks(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> np.ndarray:
    """Detect candidate R-peaks using SciPy's prominence-based peak finder."""
```
- Wrapper for R‑peak detection.

```python
    if signal.size < 4:
        return np.array([], dtype=int)
```
- If too short, returns empty array (no peaks).

```python
    distance = max(int(R_PEAK_DISTANCE_SECONDS * sampling_rate), 1)
    prominence = max(np.std(signal) * R_PEAK_PROMINENCE_SCALE, 0.1)
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
```
- `distance`: minimal index separation between peaks.
- `prominence`: height threshold relative to signal standard deviation.
- `find_peaks`: returns indices of peaks.

```python
    if peaks.size == 0:
        return np.array([int(np.argmax(signal))], dtype=int)
    return peaks.astype(int)
```
- If no peaks are found, fallback: treat global maximum as a single R‑peak.
- Otherwise, return detected peaks as integers.

```python
def extract_aligned_beats(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Center beats around each R-peak to create phase-synchronized segments."""
```
- Extracts and aligns beats around each R‑peak.

```python
    pre = max(int(HEARTBEAT_PRE_SECONDS * sampling_rate), 1)
    post = max(int(HEARTBEAT_POST_SECONDS * sampling_rate), 1)
    window = pre + post
```
- Converts pre/post times into sample counts.
- Total window length `window`.

```python
    peaks = detect_r_peaks(signal, sampling_rate)
    beats: List[np.ndarray] = []
    kept_peaks: List[int] = []
```
- Detects peaks.
- Initializes lists to store aligned beats and associated R‑peak indices.

```python
    for peak in peaks:
        start = peak - pre
        end = peak + post
        if start < 0 or end >= signal.size:
            continue
        beat = signal[start:end].astype(np.float64)
        beat -= beat.mean()
        beats.append(beat)
        kept_peaks.append(peak)
```
- For each peak:
  - Computes window bounds.
  - Skips if window would go out of signal bounds.
  - Extracts segment, casts to `float64`, subtracts mean to center.
  - Appends beat and its R‑peak index.

```python
    if not beats:
        fallback = signal[:window]
        if fallback.size < window:
            fallback = np.pad(fallback, (0, window - fallback.size))
        beats = [fallback - np.mean(fallback)]
        kept_peaks = [int(np.argmax(signal))]
```
- If no valid beats were collected:
  - Takes first `window` samples (padding if needed).
  - Centers by removing mean.
  - Uses max index as R‑peak index.

```python
    return np.vstack(beats), np.asarray(kept_peaks, dtype=int)
```
- Stacks beats into 2D array (rows = beats).
- Returns beats and peak indices.

```python
def build_feature_matrix(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create spectral feature matrix from heartbeat-aligned FFT magnitudes."""
```
- Main feature extraction function.

```python
    beats, r_peaks = extract_aligned_beats(signal, sampling_rate)
    window = np.hamming(beats.shape[1])
    windowed = beats * window
```
- Aligns beats.
- Builds Hamming window of same length as beats.
- Applies it element‑wise to reduce spectral leakage.

```python
    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    max_bins = min(HEARTBEAT_SPECTRAL_BINS, spectra.shape[1])
    spectra = spectra[:, :max_bins]
```
- Computes real FFT along time axis (rows = beats).
- Keeps only up to `HEARTBEAT_SPECTRAL_BINS` frequency bins.

```python
    norms = np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-12
    features = spectra / norms
    return features, beats, r_peaks
```
- Row‑normalize each beat spectrum to unit norm (avoid division by zero using small `1e-12`).
- Returns:
  - `features`: normalized spectral matrix.
  - `beats`: time‑domain aligned beats.
  - `r_peaks`: R‑peak indices.

```python
def run_pca(feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run PCA and return transformed coordinates plus explained variance."""
```
- Encapsulates PCA logic.

```python
    rows = feature_matrix.shape[0]
    if rows < 2:
        feature_matrix = np.vstack([feature_matrix, feature_matrix])
```
- Ensure at least 2 samples; PCA needs more than 1 observation.
  - Duplicates the single row if necessary.

```python
    pca = PCA(n_components=2, whiten=True, random_state=42)
    components = pca.fit_transform(feature_matrix)
    return components, pca.explained_variance_ratio_
```
- Creates `PCA` object:
  - `n_components=2`: 2D projection.
  - `whiten=True`: scales components to unit variance.
  - `random_state=42`: reproducible randomness.
- `fit_transform`:
  - Learns principal components and applies projection.
- Returns:
  - `components`: 2D coordinates.
  - `explained_variance_ratio_`: fraction of variance per component.

```python
def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode Matplotlib figure as base64 data URI."""
```
- Helper to convert Matplotlib figures into base64 PNG strings.

```python
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    import base64
```
- Creates in‑memory buffer.
- Tightens layout.
- Saves figure as PNG into buffer.
- Closes figure to free memory.
- Rewinds buffer.
- Imports `base64` locally.

```python
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
```
- Base64‑encodes PNG bytes, decodes to ASCII string.
- Prefixes with data URI header for embedding in HTML `<img>` tags.

```python
def create_time_plot(signal: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> str:
    """Return base64 image for time-domain ECG plot."""
```
- Plots ECG in time domain.

```python
    times = np.arange(signal.size) / sampling_rate
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, signal, color="#2563eb", linewidth=1)
    ax.set_title("ECG Signal (Time Domain)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)
```
- Computes time axis.
- Creates subplot figure and axes.
- Plots signal with labels and grid.
- Converts figure to base64 string.

```python
def create_frequency_plot(fft_res: FFTResult) -> str:
    """Return base64 image for FFT magnitude plot."""
```
- Frequency‑domain plot for full signal.

```python
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(fft_res.frequencies, fft_res.magnitudes, color="#16a34a", linewidth=1)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, fft_res.frequencies.max())
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)
```
- Simple line plot: magnitude vs frequency, limited to 0..max frequency.

```python
def create_aligned_heartbeat_plot(beats: np.ndarray, sampling_rate: float = DEFAULT_SAMPLING_RATE) -> str:
    """Return base64 image showing aligned heartbeats plus their template."""
```
- Visualizes multiple aligned beats and their average template.

```python
    if beats.size == 0:
        raise ValueError("No beats available for visualization.")
```
- Guard: requires at least one beat.

```python
    pre = max(int(HEARTBEAT_PRE_SECONDS * sampling_rate), 1)
    total_samples = beats.shape[1]
    times = (np.arange(total_samples) - pre) / sampling_rate
    fig, ax = plt.subplots(figsize=(6, 3))
```
- Recomputes pre‑window in samples.
- `times`: axis centered around 0 at R‑peak.

```python
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
```
- `subset`: optionally limit to `MAX_BEATS_FOR_PLOT` beats.
- Plots all beats in light color.
- Overlays their mean as a “template”.
- Draws vertical line at 0 (R‑peak).
- Adds labels and legend, converts to base64.

```python
def create_pca_plot(components: np.ndarray) -> str:
    """Return base64 image visualizing top 2 PCA components."""
```
- PCA scatter plot.

```python
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(components[:, 0], components[:, 1], c=np.linspace(0, 1, components.shape[0]), cmap="viridis")
    ax.set_title("PCA on Spectral Features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)
```
- Creates scatter plot of PC1 vs PC2.
- Colors points along a gradient (using `viridis` colormap).
- Returns base64 PNG.

---

## 3. `schemas.py` – Pydantic Models

```python
"""Pydantic schemas for FastAPI responses."""
```
- Short module docstring.

```python
from typing import List
from pydantic import BaseModel, Field
```
- Imports `List` for type hints, `BaseModel` and `Field` for defining models.

```python
class PlotPayload(BaseModel):
    """Container for base64 encoded plot images."""

    title: str
    image_base64: str = Field(..., description="PNG image encoded as base64 data URI.")
```
- `PlotPayload`:
  - `title`: plot name.
  - `image_base64`: base64 string for PNG; field has extra description metadata.

```python
class FormulaPayload(BaseModel):
    """Math formula metadata."""

    title: str
    latex: str
```
- `FormulaPayload`:
  - `title`: formula name.
  - `latex`: LaTeX string representing the formula.

```python
class StepPayload(BaseModel):
    """Step-by-step descriptions of the math pipeline."""

    title: str
    details: str
```
- `StepPayload`:
  - For timeline entries in the UI.

```python
class AnalysisResponse(BaseModel):
    """Full response returned after processing an ECG upload."""

    fft_frequencies: List[float]
    fft_magnitudes: List[float]
    pca_components: List[List[float]]
    explained_variance: List[float]
    plots: List[PlotPayload]
    formulas: List[FormulaPayload]
    steps: List[StepPayload]
    heartbeat_count: int
    r_peak_indices: List[int]
    method_summary: str
```
- `AnalysisResponse`:
  - Main schema for `/api/analyze`.
  - Includes:
    - FFT data.
    - PCA components and explained variance.
    - Plots, formulas, pipeline steps.
    - Heartbeat statistics.
    - Text summary of the method.

---

## 4. `math_content.py` – LaTeX Formulas

```python
"""Reference LaTeX formulas displayed by the frontend."""
```
- Docstring explaining purpose: serve pre‑written LaTeX for frontend MathJax.

```python
FOURIER_FORMULA = r"""
X(k) = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi kn / N}
"""
```
- **DFT definition**: sum over time samples with complex exponential.

```python
INVERSE_FFT_FORMULA = r"""
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X(k)\, e^{j 2\pi kn / N}
"""
```
- **Inverse DFT**: reconstruct time‑domain signal from frequency bins.

```python
ADAPTIVE_ALIGNMENT_FORMULA = r"""
\tilde{x}_k[n] = x\big(r_k + n - n_0\big), \quad -n_0 \le n < n_1
"""
```
- **Heartbeat alignment**: definition of beat segment around R‑peak.

```python
PCA_VARIANCE_FORMULA = r"""
\text{Var}(\mathbf{z}) = \lambda = \max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{S}\, \mathbf{w}
"""
```
- **PCA variance maximization**:
  - Shows that principal component maximizes variance along direction \(\mathbf{w}\).

```python
PCA_PROJECTION_FORMULA = r"""
\mathbf{z} = \mathbf{X}_c \mathbf{W}, \quad
\mathbf{X}_c = \mathbf{X} - \mathbf{1}\mu^\top
"""
```
- **PCA projection**:
  - Center data: \(\mathbf{X}_c = \mathbf{X} - \mathbf{1}\mu^\top\).
  - Project onto principal components: \(\mathbf{z} = \mathbf{X}_c \mathbf{W}\).

```python
FORMULA_BLOCKS = [
    {"title": "Discrete Fourier Transform (DFT)", "latex": FOURIER_FORMULA},
    {"title": "Inverse DFT", "latex": INVERSE_FFT_FORMULA},
    {"title": "Heartbeat Alignment", "latex": ADAPTIVE_ALIGNMENT_FORMULA},
    {"title": "PCA Variance Maximization", "latex": PCA_VARIANCE_FORMULA},
    {"title": "PCA Projection", "latex": PCA_PROJECTION_FORMULA},
]
```
- List of dictionaries used by:
  - Backend: to include formulas in `AnalysisResponse`.
  - Frontend: to display equations with MathJax.

---

## 5. `run_cli.py` – Command‑Line Interface

```python
"""Command-line entrypoint to run ECG analysis and print results to the terminal.
...
"""
```
- Module docstring:
  - Explains purpose: run analysis from terminal and print results.
  - Shows example commands.

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .math_content import FORMULA_BLOCKS
from .processing import build_feature_matrix, compute_fft, load_signal_from_csv, run_pca
from .main import _build_steps
```
- Imports:
  - `argparse`: for command‑line arguments.
  - `Path`: filesystem path handling.
  - `numpy`: for `argmax`.
  - Reuses:
    - `FORMULA_BLOCKS`: LaTeX for printing.
    - Processing helpers: same maths as API.
    - `_build_steps`: same textual step descriptions as API.

```python
def analyze_csv(csv_path: Path) -> None:
    """Run the ECG pipeline on a CSV file and print results."""
```
- Main worker function:
  - Takes a path to a CSV file.
  - Runs complete analysis.
  - Prints outputs to stdout.

```python
    if not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")
```
- Validates that the path exists and is a file.
- Uses `SystemExit` to cleanly stop if it doesn’t.

```python
    file_bytes = csv_path.read_bytes()
    signal = load_signal_from_csv(file_bytes)

    print(f"\nLoaded signal from: {csv_path}")
    print(f"Total samples: {signal.size}")
```
- Reads file bytes.
- Converts them to NumPy array using same CSV loader.
- Prints basic info.

```python
    # Core computations
    fft_res = compute_fft(signal)
    feature_matrix, beats, r_peaks = build_feature_matrix(signal)
    pca_components, variance = run_pca(feature_matrix)
```
- Identical math operations as in API:
  - Full‑signal FFT.
  - Beat‑level spectral features.
  - PCA.

```python
    # Summary metrics
    dominant_idx = int(np.argmax(fft_res.magnitudes))
    dominant_freq = float(fft_res.frequencies[dominant_idx])
    dominant_mag = float(fft_res.magnitudes[dominant_idx])
```
- Computes **dominant frequency**:
  - Find index of maximum magnitude.
  - Read corresponding frequency and magnitude.

```python
    heartbeat_count = int(len(r_peaks))
    steps = _build_steps(
        signal_len=signal.size,
        feature_rows=feature_matrix.shape[0],
        heartbeat_count=heartbeat_count,
    )
```
- `heartbeat_count`: number of detected beats.
- `steps`: list of step descriptions, same as frontend.

```python
    method_summary = (
        "Adaptive Heartbeat-Aligned Fourier PCA centers each detected heartbeat at its R-peak, "
        "applies the Fourier transform per beat to capture spectral envelopes, and runs PCA on the "
        "resulting frequency matrix to obtain low-noise embeddings."
    )
```
- Same method summary as `main.py`.

```python
    # Print pipeline summary
    print("\n=== Pipeline Summary ===")
    print(method_summary)
```
- Prints method description block.

```python
    print("\n=== Steps ===")
    for idx, step in enumerate(steps, start=1):
        print(f"{idx}. {step['title']}: {step['details']}")
```
- Prints each step with index and details.

```python
    # Numeric outputs
    print("\n=== Numeric Outputs ===")
    print(f"Heartbeat count: {heartbeat_count}")
    print(f"R-peak indices (first 20): {r_peaks[:20].tolist()}{' ...' if len(r_peaks) > 20 else ''}")
    print(f"Dominant FFT frequency: {dominant_freq:.4f} Hz")
    print(f"Dominant FFT magnitude: {dominant_mag:.6f}")
    print(
        "Explained variance (PCA components): "
        + " / ".join(f"{ratio * 100:.2f}%" for ratio in variance)
    )
    print(f"PCA components shape: {pca_components.shape}")
```
- Prints:
  - Total number of beats.
  - First 20 R‑peak indices (with `...` if more).
  - Dominant FFT frequency & magnitude.
  - PCA explained variance (%) per component.
  - Shape of PCA component matrix.

```python
    # Formulas
    print("\n=== Governing Equations (LaTeX) ===")
    for block in FORMULA_BLOCKS:
        print(f"- {block['title']}:")
        # Strip leading/trailing whitespace/newlines to keep output compact
        latex = block["latex"].strip()
        print(f"  {latex}\n")
```
- Prints all LaTeX formulas:
  - For each block, prints title and formula text.

```python
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ECG FFT + PCA analysis on a CSV file.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to ECG CSV file (single column of voltage samples).",
    )
    return parser.parse_args(argv)
```
- Builds CLI argument parser:
  - Single required `--csv` option for file path.

```python
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path = Path(args.csv).expanduser().resolve()
    analyze_csv(csv_path)
```
- `main`:
  - Parses arguments.
  - Converts CSV path string to resolved `Path`.
  - Calls `analyze_csv`.

```python
if __name__ == "__main__":
    main()
```
- Standard Python entrypoint guard:
  - When run as a script (`python app/run_cli.py`), execute `main()`.


