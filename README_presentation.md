## ECG Fourier + PCA Project – PPT Outline (Math-Focused)

This README gives you **slide-by-slide content** you can directly convert into a PowerPoint.  
Focus: **90% on maths**, with simple wording for oral explanation.

---

## Slide 1 – Title

- **Title**: ECG Signal Analysis using Fourier Transform and PCA  
- **Subtitle**: A mathematical pipeline for heartbeat pattern visualization  
- **By**: *Your Name, Your Institution*  

---

## Slide 2 – Problem Statement

- **Problem**  
  - ECG is a **discrete-time signal** \(x[n]\) with noise and large length.  
  - Doctors must compare **many heartbeats** to detect abnormal patterns.  
  - Raw time-domain ECG is hard to analyze visually and mathematically.  
- **Need**  
  - A **mathematical method** to:
    - Align individual heartbeats,
    - Convert them to the **frequency domain**,
    - Reduce them to a **low-dimensional space** for easy visualization.

---

## Slide 3 – Objectives

- **Objective 1**: Model ECG as a discrete-time signal and extract **aligned heartbeats** around R-peaks.  
- **Objective 2**: Use the **Discrete Fourier Transform (DFT)** to convert each heartbeat from time domain to frequency domain.  
- **Objective 3**: Build a **spectral feature matrix** and apply **Principal Component Analysis (PCA)** to reduce dimensionality.  
- **Objective 4**: Interpret the PCA space to understand **variance** and **frequency patterns** between heartbeats.  

---

## Slide 4 – Literature Review (Intro)

- **Goal of this slide**: Show that similar ideas exist and our work combines them.  
- **Key themes in literature**  
  - R-peak detection in ECG.  
  - Time-domain and frequency-domain features.  
  - Use of PCA and clustering for ECG beats.  

You will present the detailed table in the next slides.

---

## Slide 5 – Literature Review Table (1/2)

Create a table in PPT with columns: **S.No, Author & Year, Method / Dataset, Main Math Concept, Key Idea**.  
Fill with entries like:

1. **Pan & Tompkins, 1985** – QRS detection – *Digital filtering, thresholds* – R-peak detection using bandpass filters and slope/amplitude rules.  
2. **Moody & Mark, 1982** – MIT-BIH arrhythmia database – *Statistical analysis* – Standard dataset and annotations for ECG research.  
3. **Laguna et al., 1994** – HRV analysis – *Time-domain statistics* – Uses mean, variance of RR intervals as features.  
4. **Clifford et al., 2006** – ECG processing review – *Filtering, wavelets, DFT* – Overview of mathematical tools for ECG analysis.  
5. **Osowski & Linh, 2001** – ECG classification – *FFT features + neural nets* – Uses frequency spectra of beats as input features.  

---

## Slide 6 – Literature Review Table (2/2)

Continue the same table:

6. **Lagerholm et al., 2000** – Beat clustering – *PCA + k-means* – PCA on ECG beats followed by clustering.  
7. **Martis et al., 2013** – Feature extraction – *Wavelet transform, energy* – Uses transform-domain energies as discriminative features.  
8. **Moody & Mark, 1990s** – HRV spectral tools – *DFT of RR intervals* – Separates low- and high-frequency components.  
9. **Ince et al., 2009** – ECG compression – *Transform coding, PCA* – Uses PCA to reduce redundancy in ECG for compression.  
10. **Acharya et al., 2004** – Automated diagnosis – *Stat + transform + PCA features* – Combines multiple feature types for classification.  

**Connection sentence for you to say:**  
“Our work combines ideas from FFT-based features, PCA, and beat alignment to build a simple mathematical pipeline.”

---

## Slide 7 – Methodology Overview (Math-Only)

- **Input**: ECG discrete-time signal \(x[n]\), sampled at frequency \(f_s\).  
- **Step 1**: Detect R-peaks and cut signal into **heartbeats**.  
- **Step 2**: For each heartbeat, compute **DFT** to get frequency-domain features.  
- **Step 3**: Build a **feature matrix** from spectra of all beats.  
- **Step 4**: Apply **PCA** to project features into 2D space.  

This slide is like a roadmap; details come in the next slides.

---

## Slide 8 – ECG as Discrete-Time Signal

- **Model**  
  - ECG is represented as:
    \[
    x[n], \quad n = 0, 1, \dots, N-1
    \]
  - Sampling frequency:
    \[
    f_s \ (\text{samples per second})
    \]
- **Goal**  
  - From a long sequence \(x[n]\), we want to extract many **short segments** around each heartbeat.  
- **Reason**  
  - Working with **aligned short signals** makes both Fourier analysis and PCA easier and more meaningful.

---

## Slide 9 – R-Peak Detection and Heartbeat Alignment

- **R-peak set**
  - Indices of R-peaks:
    \[
    \mathcal{R} = \{r_1, r_2, \dots, r_K\}
    \]
  - R-peaks are found using **amplitude threshold + minimum distance** between peaks.  
- **Alignment window**
  - Choose pre-window \(n_0\) and post-window \(n_1\) (in samples).  
  - For each peak \(r_k\), define an aligned heartbeat:
    \[
    \tilde{x}_k[n] = x(r_k + n), \quad -n_0 \le n < n_1
    \]
- **Result**  
  - All beats \(\tilde{x}_k[n]\) have the **same length** \(L = n_0 + n_1\), centered at the R-peak.

---

## Slide 10 – Fourier Transform per Heartbeat

- **Discrete Fourier Transform (DFT)** for beat \(k\):
  \[
  X_k(m) = \sum_{n=0}^{L-1} \tilde{x}_k[n] \, e^{-j 2\pi mn / L}, \quad m = 0, 1, \dots, L-1
  \]
- **Magnitude spectrum**:
  \[
  |X_k(m)| = \sqrt{\Re(X_k(m))^2 + \Im(X_k(m))^2}
  \]
- **Feature vector**  
  - Keep the first \(M\) frequency bins:
    \[
    \mathbf{f}_k = [|X_k(0)|, |X_k(1)|, \dots, |X_k(M-1)|]
    \]
- **Interpretation**  
  - Each beat is now represented by **frequency components** instead of time samples.

---

## Slide 11 – Feature Matrix and Normalization

- **Feature matrix**:
  \[
  \mathbf{F} =
  \begin{bmatrix}
  - \mathbf{f}_1 - \\
  - \mathbf{f}_2 - \\
  \vdots \\
  - \mathbf{f}_K -
  \end{bmatrix}
  \in \mathbb{R}^{K \times M}
  \]
  where each row is one heartbeat in frequency domain.  
- **Row normalization (optional)**:
  \[
  \hat{\mathbf{f}}_k = \frac{\mathbf{f}_k}{\|\mathbf{f}_k\|_2}
  \]
- **Reason**  
  - Normalization removes scale differences so PCA focuses on **shape of spectra**, not just amplitude.

---

## Slide 12 – PCA: Centering and Covariance

- **Data centering**:
  \[
  \boldsymbol{\mu} = \frac{1}{K} \sum_{k=1}^{K} \hat{\mathbf{f}}_k
  \]
  \[
  \mathbf{X}_c = \mathbf{F} - \mathbf{1}\boldsymbol{\mu}^\top
  \]
- **Covariance matrix**:
  \[
  \mathbf{S} = \frac{1}{K-1} \mathbf{X}_c^\top \mathbf{X}_c
  \]
- **Interpretation**  
  - \(\mathbf{S}\) captures how different frequency bins **co-vary** across heartbeats.

---

## Slide 13 – PCA: Eigenvalues and Projection

- **Eigen decomposition**:
  \[
  \mathbf{S}\mathbf{w}_i = \lambda_i \mathbf{w}_i
  \]
  where \(\lambda_i\) is variance in direction \(\mathbf{w}_i\).  
- **Order eigenvalues**:
  \[
  \lambda_1 \ge \lambda_2 \ge \dots
  \]
- **Principal components matrix**:
  \[
  \mathbf{W} = [\mathbf{w}_1 \ \mathbf{w}_2]
  \]
- **Projection of each beat**:
  \[
  \mathbf{z}_k = \mathbf{X}_c(k,:) \, \mathbf{W}
  \]
  So each heartbeat is now a **2D point** \((z_{k1}, z_{k2})\) that we can plot.

---

## Slide 14 – Results: Example Outputs

- **1. Dominant frequency of signal**  
  - From full-signal FFT, find frequency where \(|X(f)|\) is maximum.  
- **2. Heartbeat count and R-peak indices**  
  - Number of beats \(K\), list of R-peak positions.  
- **3. PCA explained variance**  
  - Ratios:
    \[
    \text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}
    \]
  - Example: PC1 = 70%, PC2 = 20%, others = 10%.  
- **4. PCA scatter plot**  
  - Points cluster according to heartbeat morphology (similar shapes close together).

---

## Slide 15 – Conclusion

- The proposed pipeline:
  - **Aligns** heartbeats around R-peaks,  
  - Uses **DFT** to convert each beat to a frequency-domain representation,  
  - Applies **PCA** to reduce dimensions and visualize patterns.  
- This **mathematical approach** turns complex ECG time series into **simple 2D patterns** that are easier to interpret.  
- It provides a basis for further tasks like clustering and automated diagnosis.

---

## Slide 16 – Future Scope

- Add **clustering algorithms** (e.g., k-means, Gaussian Mixture Models) in PCA space to group beat types.  
- Explore **non-linear dimensionality reduction** (t-SNE, UMAP) for more complex structures.  
- Extend the method to **multi-lead ECG** by combining features from several leads.  
- Integrate simple **classification models** (SVM, neural networks) using PCA features for arrhythmia detection.  

---

## Slide 17 – Thank You

- **Thank you**  
- **Questions?**  
- Contact: *Your email / details*  


