"""FastAPI application wiring ECG analysis helpers to HTTP endpoints."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

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

APP_NAME = "Fourier + PCA ECG Signal Analysis API"
API_PREFIX = "/api"

app = FastAPI(title=APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_steps(signal_len: int, feature_rows: int, heartbeat_count: int) -> list[dict[str, str]]:
    """Return descriptive pipeline steps for UI timeline."""
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


@app.get(f"{API_PREFIX}/health")
def healthcheck() -> dict[str, str]:
    """Lightweight readiness probe."""
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/analyze", response_model=AnalysisResponse)
async def analyze_ecg(file: UploadFile = File(...)) -> AnalysisResponse:
    """Accept ECG CSV, run FFT + PCA, and return plots plus numeric outputs."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        raw_bytes = await file.read()
        signal = load_signal_from_csv(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fft_res = compute_fft(signal)
    feature_matrix, beats, r_peaks = build_feature_matrix(signal)
    pca_components, variance = run_pca(feature_matrix)

    # Guard against NaN/inf values that cannot be JSON-encoded.
    fft_freqs_clean = np.nan_to_num(fft_res.frequencies, nan=0.0, posinf=0.0, neginf=0.0)
    fft_mags_clean = np.nan_to_num(fft_res.magnitudes, nan=0.0, posinf=0.0, neginf=0.0)
    pca_components_clean = np.nan_to_num(pca_components, nan=0.0, posinf=0.0, neginf=0.0)
    variance_clean = np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)

    method_summary = (
        "Adaptive Heartbeat-Aligned Fourier PCA centers each detected heartbeat at its R-peak, "
        "applies the Fourier transform per beat to capture spectral envelopes, and runs PCA on the "
        "resulting frequency matrix to obtain low-noise embeddings."
    )

    plots = [
        {"title": "ECG Signal (Time Domain)", "image_base64": create_time_plot(signal)},
        {"title": "FFT Magnitude Spectrum", "image_base64": create_frequency_plot(fft_res)},
        {"title": "Aligned Heartbeats", "image_base64": create_aligned_heartbeat_plot(beats)},
        {"title": "PCA Scatter Plot", "image_base64": create_pca_plot(pca_components)},
    ]

    response = AnalysisResponse(
        fft_frequencies=fft_freqs_clean.tolist(),
        fft_magnitudes=fft_mags_clean.tolist(),
        pca_components=pca_components_clean.tolist(),
        explained_variance=variance_clean.tolist(),
        plots=plots,
        formulas=FORMULA_BLOCKS,
        steps=_build_steps(signal_len=signal.size, feature_rows=feature_matrix.shape[0], heartbeat_count=int(len(r_peaks))),
        heartbeat_count=int(len(r_peaks)),
        r_peak_indices=r_peaks.tolist(),
        method_summary=method_summary,
    )
    return response


__all__ = ["app"]


