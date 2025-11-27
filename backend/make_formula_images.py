"""
Generate LaTeX formula images used in the documentation.

This script saves PNG files like:
  - formula1_prominence.png
  - formula2_interpeak.png
  - ...
  - formula14_projection.png

Run from the project root:

    cd /Users/vigneshraj/Downloads/NEW2
    python backend/make_formula_images.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE  # save images next to this script


def save_formula(filename: str, latex: str) -> None:
    """Render a single LaTeX formula into a PNG image."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 2))
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=18)
    plt.axis("off")
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    formulas = {
        "formula1_prominence.png": r"\mathrm{prominence} \geq \alpha \cdot \sigma_x",
        "formula2_interpeak.png": r"r_{k+1} - r_k \geq f_s \cdot T_{\min}",
        "formula3_wpre.png": r"n_0 = \left\lfloor f_s \cdot T_{\text{pre}} \right\rfloor",
        "formula4_wpost.png": r"n_1 = \left\lfloor f_s \cdot T_{\text{post}} \right\rfloor",
        "formula5_dcremoval.png": r"\tilde{x}_k[n] = x_k[n] - \frac{1}{L} \sum_{m=0}^{L-1} x_k[m]",
        "formula6_hamming.png": r"w[n] = 0.54 - 0.46 \cos\!\left(\frac{2\pi n}{L-1}\right),\ 0 \leq n \leq L-1",
        "formula7_fft.png": r"X_k(m) = \sum_{n=0}^{L-1} \tilde{x}_k^{(w)}[n] e^{-j 2\pi mn / L}",
        "formula8_magnitude.png": r"|X_k(m)| = \sqrt{\Re(X_k(m))^2 + \Im(X_k(m))^2}",
        "formula9_truncation.png": r"\mathbf{f}_k = [ |X_k(0)|, \dots, |X_k(127)| ]",
        "formula10_l2norm.png": r"\hat{\mathbf{f}}_k = \frac{\mathbf{f}_k}{\|\mathbf{f}_k\|_2}",
        "formula11_mean.png": r"\boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^{K} \hat{\mathbf{f}}_k",
        "formula12_centered.png": r"\mathbf{X}_c = \mathbf{F} - \mathbf{1} \boldsymbol{\mu}^{\top}",
        "formula13_covariance.png": r"\mathbf{S} = \frac{1}{K-1} \mathbf{X}_c^{\top} \mathbf{X}_c",
        "formula14_projection.png": r"\mathbf{Z} = \mathbf{X}_c \mathbf{W},\ \mathbf{W} = [\mathbf{w}_1\ \mathbf{w}_2]",
    }

    for fname, latex in formulas.items():
        print(f"Saving {fname} ...")
        save_formula(fname, latex)

    print(f"Done. Images saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()


