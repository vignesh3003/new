"""Command-line entrypoint to run ECG analysis and print results to the terminal.

Usage examples (from the project root):

    cd backend
    python -m app.run_cli --csv ../sample_ecg.csv

or with your own ECG CSV:

    python -m app.run_cli --csv /path/to/your_ecg.csv

This uses the same processing pipeline as the FastAPI service but
prints key numeric outputs and formulas directly to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .math_content import FORMULA_BLOCKS
from .processing import build_feature_matrix, compute_fft, load_signal_from_csv, run_pca
from .main import _build_steps


def analyze_csv(csv_path: Path) -> None:
    """Run the ECG pipeline on a CSV file and print results."""
    if not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    file_bytes = csv_path.read_bytes()
    signal = load_signal_from_csv(file_bytes)

    print(f"\nLoaded signal from: {csv_path}")
    print(f"Total samples: {signal.size}")

    # Core computations
    fft_res = compute_fft(signal)
    feature_matrix, beats, r_peaks = build_feature_matrix(signal)
    pca_components, variance = run_pca(feature_matrix)

    # Summary metrics
    dominant_idx = int(np.argmax(fft_res.magnitudes))
    dominant_freq = float(fft_res.frequencies[dominant_idx])
    dominant_mag = float(fft_res.magnitudes[dominant_idx])

    heartbeat_count = int(len(r_peaks))
    steps = _build_steps(
        signal_len=signal.size,
        feature_rows=feature_matrix.shape[0],
        heartbeat_count=heartbeat_count,
    )

    method_summary = (
        "Adaptive Heartbeat-Aligned Fourier PCA centers each detected heartbeat at its R-peak, "
        "applies the Fourier transform per beat to capture spectral envelopes, and runs PCA on the "
        "resulting frequency matrix to obtain low-noise embeddings."
    )

    # Print pipeline summary
    print("\n=== Pipeline Summary ===")
    print(method_summary)

    print("\n=== Steps ===")
    for idx, step in enumerate(steps, start=1):
        print(f"{idx}. {step['title']}: {step['details']}")

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

    # Formulas
    print("\n=== Governing Equations (LaTeX) ===")
    for block in FORMULA_BLOCKS:
        print(f"- {block['title']}:")
        # Strip leading/trailing whitespace/newlines to keep output compact
        latex = block["latex"].strip()
        print(f"  {latex}\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ECG FFT + PCA analysis on a CSV file.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to ECG CSV file (single column of voltage samples).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    csv_path = Path(args.csv).expanduser().resolve()
    analyze_csv(csv_path)


if __name__ == "__main__":
    main()


