"""Reference LaTeX formulas displayed by the frontend."""

FOURIER_FORMULA = r"""
X(k) = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi kn / N}
"""

INVERSE_FFT_FORMULA = r"""
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X(k)\, e^{j 2\pi kn / N}
"""

ADAPTIVE_ALIGNMENT_FORMULA = r"""
\tilde{x}_k[n] = x\big(r_k + n - n_0\big), \quad -n_0 \le n < n_1
"""

PCA_VARIANCE_FORMULA = r"""
\text{Var}(\mathbf{z}) = \lambda = \max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{S}\, \mathbf{w}
"""

PCA_PROJECTION_FORMULA = r"""
\mathbf{z} = \mathbf{X}_c \mathbf{W}, \quad
\mathbf{X}_c = \mathbf{X} - \mathbf{1}\mu^\top
"""

FORMULA_BLOCKS = [
    {"title": "Discrete Fourier Transform (DFT)", "latex": FOURIER_FORMULA},
    {"title": "Inverse DFT", "latex": INVERSE_FFT_FORMULA},
    {"title": "Heartbeat Alignment", "latex": ADAPTIVE_ALIGNMENT_FORMULA},
    {"title": "PCA Variance Maximization", "latex": PCA_VARIANCE_FORMULA},
    {"title": "PCA Projection", "latex": PCA_PROJECTION_FORMULA},
]


