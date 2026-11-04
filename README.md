# EE367 Project — WNNM Image Denoising

<!--
IMPORTANT PROJECT OVERVIEW:
This repository provides a complete Python / NumPy reproduction and benchmark suite for
Weighted Nuclear Norm Minimization (WNNM) applied to image denoising, compared against
classical spatial and non-local baseline filters (Gaussian, Bilateral, Non-Local Means),
plus six original student extensions documented in the report.

Reference Publication:
S. Gu, L. Zhang, W. Zuo and X. Feng,
"Weighted Nuclear Norm Minimization with Application to Image Denoising,"
2014 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 2862-2869.
DOI: 10.1109/CVPR.2014.366
-->

Stanford EE367 (Computational Imaging) Final Project — Implementation, Reproduction, and Extensions of Weighted Nuclear Norm Minimization (WNNM) for Image Denoising.  
Project Repository: [Image_processing](https://github.com/aayushkumar5/Image_processing.git)

---

## Repository Structure

```
EE367_Project_WNNM_denoising/
│
├── our_contribution/              ← Our original work (see below)
│   ├── WNNM_IEEE_Report (2).docx  ← Full IEEE-style project report
│   └── WNNM_Reproduction_2.ipynb  ← Python reproduction + 6 extensions
│
├── implementation_of_paper/       ← Faithful re-implementation of Gu et al.
│   ├── wnnm_final.py
│   ├── filters_hw2.py
│   ├── Results_greyscale_one_img.ipynb
│   ├── Results_greyscale_all.ipynb
│   ├── Parameter_tuning_c_n.ipynb
│   ├── Parameter_tuning_K_and_delta.ipynb
│   └── report.md
│
├── results_greyscale/             ← Generated output images and metric spreadsheets
├── test_images_greyscale/         ← Input grayscale .gif benchmark images
├── README.md
└── .gitignore
```

---

## Our Contributions (`our_contribution/`)

> **Highlight key used in the report:**  
> 🟡 Yellow = our original contribution &nbsp;|&nbsp; 🔴 Red = contribution of Gu et al. [1] and their released MATLAB code

The following contributions are documented in `WNNM_IEEE_Report (2).docx` and implemented in `WNNM_Reproduction_2.ipynb`:

### 1. Faithful Python Reproduction
- Compact **NumPy + batched SVD** implementation checked **line-by-line** against the official MATLAB code and the paper.
- Patch extraction via sliding windows; all groups processed in a single batched SVD call (removes the per-group MATLAB loop).
- Reaches **36.87 dB at σ = 10** and **32.56 dB at σ = 30** on the House image — within **0.1 dB** of the published values (36.95 dB / 32.52 dB).

### 2. Bug Fixes in the Public MATLAB Demo
Three defects found and corrected compared to the public repository:
1. Demo script added noise with σ = 40 to a \[0, 1\] image → produced meaningless output.
2. Corrected index list in block-matching was computed but never used → key patch was not guaranteed to be first in its group.
3. Hard-coded Windows paths and a swapped argument order in the PSNR call.

### 3. Numerical Check of Theorem 2
- Paper states Theorem 2 for **non-ascending** weights, but our test confirms the closed-form solution is optimal only for **ascending** weights — consistent with what the official code actually uses.
- With the data term ‖Y − X‖²_F (without ½), the exact threshold is w_i / 2; the constant `c` absorbs this factor.

### 4. Stronger NNM Baseline
- Paper compares against NNM with "the uniform weight √n σ_n" without specifying the exact constant.
- We tune c_NNM = 0.5 from {0.2, 0.3, 0.4, 0.5, 1, 1.5, 2} via a scan on two 128 × 128 crops — producing a **stronger NNM baseline** than used in the paper.

### 5. Six Original Extensions

| Extension | Result |
|:---|:---|
| **Optimal singular-value shrinkage** (Gavish & Donoho [5]) | ❌ −1.2 dB vs. WNNM (negative result) |
| **Rank-aware patch aggregation** (weight = 1 / (1 + rank X̂_j)) | ❌ −0.1 dB vs. WNNM (negative result) |
| **Orthonormal opponent colour transform** | ✅ **+1.7 dB PSNR, +0.065 SSIM** over per-channel RGB WNNM |
| **Luminance-guided patch grouping** | ✅ Same PSNR as standard WNNM, **up to 1.4× faster** |
| **Chroma-aware thresholding** (larger threshold on chroma channels) | ✅ **+0.2 dB** on average, confirmed on a held-out image |
| **Anscombe variance-stabilised (GAT) denoising** for signal-dependent noise | ✅ **+2.5 dB on average** (range: +0.4 to +5.5 dB) |

### 6. Practical Studies

- **Noise-level sensitivity:** Under-estimation hurts more than over-estimation. At 0.6× the true σ, WNNM loses **5.5 dB** vs. NNM's 2.9 dB loss — NNM becomes 2.0 dB better than WNNM under severe under-estimation.
- **Speed–quality trade-off (key-patch stride):** Stride 4 is **3.7× faster** than the default stride 2 at only −0.01 dB. Halving iterations (K = 4) saves 42% of runtime at −0.14 dB.

### 7. Noise-Level Estimator Pipeline
- Ported a blind noise-level estimator to complete the pipeline for images where σ is unknown.
- Fitted noise curve nearly matches the true curve (losing < 0.03 dB on the denoising result).

---

## Implementation of Paper (`implementation_of_paper/`)

Faithful re-implementation of Gu et al. (CVPR 2014) comparing WNNM against three classical baselines.

| File | Purpose |
|:---|:---|
| `wnnm_final.py` | Core WNNM algorithm — SVD shrinkage, iterative regularisation, patch aggregation |
| `filters_hw2.py` | Gaussian, Bilateral, and Non-Local Means filter implementations |
| `Results_greyscale_one_img.ipynb` | Side-by-side denoising demo on a single image with PSNR/SSIM |
| `Results_greyscale_all.ipynb` | Benchmark across all 15 test images; exports metrics to Excel |
| `Parameter_tuning_c_n.ipynb` | Grid search over weight scale `c` and patch count `N_threshold` |
| `Parameter_tuning_K_and_delta.ipynb` | Grid search over iteration count `K` and feedback factor `delta` |
| `report.md` | Technical summary report |

### Recommended Parameters (σ = 0.1 Gaussian noise)

| Parameter | Value | Description |
|:---|:---|:---|
| `patchRadius` | 3 | Patch size = (2×3)² = 36 px |
| `searchWindowRadius` | 9 | 19×19 search neighbourhood |
| `N_threshold` | 50 | Similar patches per group |
| `c` | 0.005 | Singular value weight scale |
| `delta` | 0.1 | Residual feedback factor |
| `K` | 3 | Outer iterations |

---

## Key Theoretical Insights

**Why WNNM outperforms NNM:**  
NNM applies a uniform soft-threshold λ to all singular values — heavily penalising large ones that encode structure and edges.  
WNNM assigns inversely proportional, non-descending weights:

$$w_i = \frac{c\sqrt{m}}{\hat{\sigma}_i + \varepsilon}$$

Small singular values (noise) receive large penalties; large ones (structure) are mostly preserved.

**Iterative Regularisation:**  
At each outer iteration k, the input is updated as:

$$Y^{(k)} = \hat{X}^{(k-1)} + \delta \left(Y - \hat{X}^{(k-1)}\right)$$

Re-injecting a fraction δ of the residual noise prevents fine texture loss over successive shrinkage steps.

---

## Dependencies

```
numpy
scipy
scikit-image
matplotlib
pandas
openpyxl
```

---

## References

1. S. Gu, L. Zhang, W. Zuo, and X. Feng, "Weighted nuclear norm minimization with application to image denoising," *CVPR*, 2014. DOI: [10.1109/CVPR.2014.366](https://doi.org/10.1109/CVPR.2014.366)
2. K. Dabov et al., "Image denoising by sparse 3-D transform-domain collaborative filtering," *IEEE TIP*, 2007.
3. A. Buades, B. Coll, and J.-M. Morel, "A non-local algorithm for image denoising," *CVPR*, 2005.
4. M. Gavish and D. L. Donoho, "Optimal shrinkage of singular values," *IEEE Trans. Inf. Theory*, 2017.
5. M. Mäkitalo and A. Foi, "Optimal inversion of the generalized Anscombe transformation," *IEEE TIP*, 2013.
