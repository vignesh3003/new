## Full Mathematical Explanation – ECG Fourier + PCA Pipeline

This document focuses only on the **math** behind the project.  
You can export it as a separate PDF for “math theory”.

---

## 1. ECG as a Discrete-Time Signal

- We model an ECG recording as a **discrete-time sequence**:
  \[
  x[n], \quad n = 0,1,\dots,N-1
  \]
- Here:
  - \(x[n]\): voltage (in mV) at sample index \(n\).
  - \(N\): total number of samples.
  - Sampling frequency \(f_s\) (samples per second), e.g. \(f_s = 360\text{ Hz}\).
- Relationship between **index** and **time**:
  \[
  t_n = \frac{n}{f_s}
  \]

---

## 2. R-Peak Detection (Beat Locations)

We want to locate the main spikes (R-peaks) in the ECG.

- Let:
  - \(\mathcal{R} = \{ r_1, r_2, \dots, r_K \}\) be the set of indices where R-peaks occur.
- Basic constraints for valid peaks:
  1. **Minimum distance** between peaks:
     \[
     r_{k+1} - r_k \geq d = f_s \cdot T_{\min}
     \]
     where \(T_{\min}\) is the minimum allowed RR interval (e.g. \(0.2\) seconds).
  2. **Prominence threshold**: a peak at index \(r_k\) should be big compared to noise.  
     A simple model:
     \[
     \text{prominence} \geq \alpha \cdot \sigma_x
     \]
     where:
     - \(\sigma_x\) is the standard deviation of \(x[n]\),
     - \(\alpha\) is a scaling factor (e.g. \(0.6\)).

If no peaks satisfy these conditions, we can fall back to:
\[
r_1 = \arg\max_{0 \leq n < N} x[n]
\]

---

## 3. Heartbeat Alignment Around Each R-Peak

For each R-peak \(r_k\), we cut out a small window around it.

- Choose:
  - Pre-window duration \(T_\text{pre}\) (seconds),
  - Post-window duration \(T_\text{post}\) (seconds).
- Convert to samples:
  \[
  n_0 = \lfloor f_s \cdot T_\text{pre} \rfloor, \quad
  n_1 = \lfloor f_s \cdot T_\text{post} \rfloor
  \]
  Total length:
  \[
  L = n_0 + n_1
  \]

- Define the **aligned heartbeat** for beat \(k\):
  \[
  \tilde{x}_k[n] = x(r_k + n), \quad -n_0 \le n < n_1
  \]

- In array form, we commonly index from \(0\) to \(L-1\), but conceptually:
  - \(n = 0\) corresponds to the R-peak.
  - Negative \(n\) values are before the R-peak.
  - Positive \(n\) values are after the R-peak.

- We also remove the mean to center each beat:
  \[
  \tilde{x}_k[n] \leftarrow \tilde{x}_k[n] - \frac{1}{L}\sum_{m=-n_0}^{n_1-1} \tilde{x}_k[m]
  \]

Result: we get \(K\) aligned beats, each of length \(L\).

---

## 4. Windowing and Discrete Fourier Transform (DFT)

### 4.1 Hamming Window

To reduce spectral leakage, we apply a **Hamming window** to each beat.

- Hamming window \(w[n]\) of length \(L\):
  \[
  w[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{L - 1}\right), \quad n = 0,\dots,L-1
  \]
- Windowed beat:
  \[
  \tilde{x}_k^{(w)}[n] = \tilde{x}_k[n] \cdot w[n]
  \]

### 4.2 DFT of Each Beat

For each beat, we compute the **Discrete Fourier Transform (DFT)**:

- For beat \(k\):
  \[
  X_k(m) = \sum_{n=0}^{L-1} \tilde{x}_k^{(w)}[n]\; e^{-j 2\pi \frac{mn}{L}},
  \quad m = 0,1,\dots,L-1
  \]

- The complex value \(X_k(m)\) can be written as:
  \[
  X_k(m) = A_k(m) + j B_k(m)
  \]
  where \(A_k(m) = \Re(X_k(m))\) and \(B_k(m) = \Im(X_k(m))\).

- We are mainly interested in the **magnitude spectrum**:
  \[
  |X_k(m)| = \sqrt{A_k(m)^2 + B_k(m)^2}
  \]

Because the ECG is real-valued, the spectrum is symmetric, so we keep only the **first half** of the frequencies.

---

## 5. Frequency-Domain Feature Vector for Each Beat

We convert each beat \(k\) into a **feature vector** using the magnitude of its DFT.

- Define the number of frequency bins to keep:
  \[
  M = \min(M_\text{max}, L')
  \]
  where:
  - \(M_\text{max}\) is a design choice (e.g., 128 bins),
  - \(L'\) is the number of unique positive frequency bins from the FFT (roughly \(L/2\)).

- Feature vector for beat \(k\):
  \[
  \mathbf{f}_k =
  \big[
    |X_k(0)|,\,
    |X_k(1)|,\,
    \dots,\,
    |X_k(M-1)|
  \big]
  \in \mathbb{R}^{M}
  \]

- To remove scale dependence, we often **normalize** each vector:
  \[
  \|\mathbf{f}_k\|_2
  = \sqrt{\sum_{m=0}^{M-1} |X_k(m)|^2}
  \]
  \[
  \hat{\mathbf{f}}_k = \frac{\mathbf{f}_k}{\|\mathbf{f}_k\|_2 + \varepsilon}
  \]
  where \(\varepsilon > 0\) is a tiny constant to avoid division by zero.

---

## 6. Feature Matrix Across All Beats

Stack the normalized feature vectors into a **feature matrix**:

- \[
  \mathbf{F} =
  \begin{bmatrix}
  - \hat{\mathbf{f}}_1 - \\
  - \hat{\mathbf{f}}_2 - \\
  \vdots \\
  - \hat{\mathbf{f}}_K -
  \end{bmatrix}
  \in \mathbb{R}^{K \times M}
  \]
- Each row represents one heartbeat in the frequency domain.
- Purpose:
  - Transform many beats into a structured dataset ready for PCA and other machine learning.

---

## 7. Centering and Covariance for PCA

### 7.1 Centering the Data

PCA assumes data is **mean-centered**.

- Compute the mean feature vector:
  \[
  \boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^{K} \hat{\mathbf{f}}_k
  \]
- Center the data:
  \[
  \mathbf{X}_c(k,:) = \hat{\mathbf{f}}_k - \boldsymbol{\mu}
  \]
- In matrix form:
  \[
  \mathbf{X}_c = \mathbf{F} - \mathbf{1}\boldsymbol{\mu}^\top
  \]
  where:
  - \(\mathbf{1}\) is a \(K \times 1\) column vector of ones.
  - \(\boldsymbol{\mu}^\top\) is a row vector (1 × M).

### 7.2 Covariance Matrix

The sample **covariance matrix** of the features is:

\[
\mathbf{S} = \frac{1}{K - 1} \mathbf{X}_c^\top \mathbf{X}_c
\]

- \(\mathbf{S}\) is an \(M \times M\) symmetric matrix.
- Each entry \(S_{ij}\) measures how frequency bin \(i\) and bin \(j\) vary together across beats.

---

## 8. PCA: Eigen-Decomposition

To perform PCA, we solve the **eigenvalue problem**:

\[
\mathbf{S} \mathbf{w}_i = \lambda_i \mathbf{w}_i
\]

- Here:
  - \(\lambda_i\): eigenvalue (variance in direction \(\mathbf{w}_i\)).
  - \(\mathbf{w}_i\): eigenvector (direction in feature space).

We order eigenvalues:

\[
\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_M \geq 0
\]

The first **principal component** is the direction with:

\[
\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|_2 = 1} \mathbf{w}^\top \mathbf{S} \mathbf{w}
\]

This is the same as the formula:

\[
\text{Var}(\mathbf{z}) = \lambda
= \max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{S}\, \mathbf{w}
\]

which appears in your `math_content.py`.

---

## 9. Dimensionality Reduction and Projection

To visualize the data, we keep only the **top two** principal components.

- Let:
  \[
  \mathbf{W} = [\mathbf{w}_1\ \mathbf{w}_2]
  \]
  be the matrix whose columns are the first two eigenvectors.

- Projection of the centered data:
  \[
  \mathbf{Z} = \mathbf{X}_c \mathbf{W}
  \]

- Each row \(\mathbf{z}_k\) of \(\mathbf{Z}\) is:
  \[
  \mathbf{z}_k =
  \begin{bmatrix}
  z_{k1} & z_{k2}
  \end{bmatrix}
  \]

This gives us a **2D point for each beat**, which we can plot on a scatter plot.

---

## 10. Explained Variance Ratios

The eigenvalues are directly related to the **variance** captured by each principal component:

- Total variance:
  \[
  \text{Var}_\text{total} = \sum_{i=1}^{M} \lambda_i
  \]

- Explained variance ratio of component \(i\):
  \[
  \text{EVR}_i = \frac{\lambda_i}{\text{Var}_\text{total}}
  \]

If we keep 2 components, we might have, for example:

- \(\text{EVR}_1 = 0.70\) (70%),
- \(\text{EVR}_2 = 0.20\) (20%).

So the first two components capture **90%** of the total variance.

---

## 11. Time-Domain and Frequency-Domain FFT of the Full Signal

Besides beat-level FFTs, we also compute an FFT on the **entire ECG signal**:

- DFT of full signal:
  \[
  X(m) = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi \frac{mn}{N}}, \quad m = 0,\dots,N-1
  \]

- Frequency axis:
  \[
  f_m = \frac{m}{N} f_s
  \]

- Single-sided magnitude spectrum (only non-negative frequencies):
  \[
  |X(m)|_\text{single} =
    \frac{2}{N} \left| X(m) \right|, \quad m = 0,\dots,\left\lfloor \frac{N}{2} \right\rfloor
  \]

- **Dominant frequency**:
  \[
  m^\* = \arg\max_m |X(m)|_\text{single}
  \]
  \[
  f^\* = f_{m^\*}
  \]

This \(f^\*\) is the frequency where the spectrum has maximum energy (e.g. related to heart rate or its harmonics).

---

## 12. Summary of the Mathematical Pipeline

1. **Model ECG** as discrete-time signal \(x[n]\) with sampling frequency \(f_s\).  
2. **Detect R-peaks** to find indices \(\mathcal{R} = \{r_k\}\).  
3. For each \(r_k\), **align a fixed-length segment** around the R-peak to get \(\tilde{x}_k[n]\).  
4. **Window** each beat with a Hamming window \(w[n]\).  
5. Compute the **DFT** of each windowed beat and extract magnitude spectra \(|X_k(m)|\).  
6. Form a **feature vector** \(\mathbf{f}_k\) from the first \(M\) magnitude bins, then normalize it.  
7. Stack features into a **feature matrix** \(\mathbf{F} \in \mathbb{R}^{K\times M}\).  
8. **Center** data and compute **covariance matrix** \(\mathbf{S}\).  
9. Perform **PCA** via eigen-decomposition of \(\mathbf{S}\), obtaining eigenvalues \(\lambda_i\) and eigenvectors \(\mathbf{w}_i\).  
10. Keep top 2 components, form \(\mathbf{W}\), and project to \(\mathbf{Z} = \mathbf{X}_c \mathbf{W}\).  
11. Use \(\mathbf{Z}\) to visualize beats in 2D and compute **explained variance ratios**.  

This gives a complete **mathematical story** from raw ECG samples to a 2D representation of heartbeat patterns.







