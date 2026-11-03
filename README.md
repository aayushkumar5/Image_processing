# EE367_Project_WNNM_denoising

<!-- 
IMPORTANT PROJECT OVERVIEW:
This repository provides a complete Python / NumPy reproduction and benchmark suite for
Weighted Nuclear Norm Minimization (WNNM) applied to image denoising, compared against
classical spatial and non-local baseline filters (Gaussian, Bilateral, Non-Local Means).

Reference Publication:
S. Gu, L. Zhang, W. Zuo and X. Feng, 
"Weighted Nuclear Norm Minimization with Application to Image Denoising," 
2014 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 2862-2869.
DOI: 10.1109/CVPR.2014.366
-->

Stanford EE367 (Computational Imaging) Final Project: Implementation and Empirical Analysis of Weighted Nuclear Norm Minimization for Image Denoising.  
Original Author: Andrei Kanavalau  
Project Repository: [Image_processing](https://github.com/aayushkumar5/Image_processing.git)

---

## Key Theoretical Insights

1. **Why WNNM outperforms Nuclear Norm Minimization (NNM):**
   * Traditional NNM applies a uniform soft-threshold ($\lambda$) to all singular values. This heavily penalizes large singular values which represent major structural and edge components, leading to over-smoothing.
   * WNNM assigns inversely proportional, non-descending weights ($w_i = c\sqrt{m} / (\sigma_i + \epsilon)$) to singular values, heavily penalizing small singular values (which capture noise) while largely preserving major structural singular values.

2. **Iterative Regularization:**
   * At each outer iteration $k$, the input is adjusted as $Y^{(k)} = \hat{X}^{(k-1)} + \delta(Y - \hat{X}^{(k-1)})$.
   * Adding back a fraction $\delta$ of the residual noise prevents fine texture loss over successive shrinkage steps.

3. **Non-Local Patch Self-Similarity:**
   * Groups $N_{threshold}$ similar patches across a local search window into a matrix $Y_j \in \mathbb{R}^{d \times m}$.
   * Subtracting patch means focuses the low-rank SVD factorization on texture fluctuations rather than baseline DC luminance.

---

## Repository Structure & Code Overview

| File / Directory | Purpose & Important Details |
| :--- | :--- |
| `wnnm_final.py` | Core implementation of WNNM denoising with SVD shrinkage, iterative regularization, and patch aggregation. |
| `filters_hw2.py` | Classical spatial & non-local filter implementations: 2D Gaussian filter, Bilateral filter, and Non-Local Means (NLM). |
| `Results_greyscale_one_img.ipynb` | Interactive demonstration applying Gaussian, Bilateral, NLM, and WNNM to a single grayscale test image (`12.gif`) with visual side-by-side comparison. |
| `Results_greyscale_all.ipynb` | Comprehensive benchmark evaluating all 4 denoising methods across 15 standard grayscale images, recording PSNR and SSIM into Excel spreadsheets. |
| `Parameter_tuning_c_n.ipynb` | Systematic grid search tuning weight scale $c \in \{0.001, 0.005, 0.01\}$ and patch threshold $N_{threshold} \in [30, 100]$. |
| `Parameter_tuning_K_and_delta.ipynb` | Systematic grid search tuning outer iteration count $K \in \{1, 3, 5\}$ and regularization parameter $\delta \in \{0.05, 0.1, 0.25\}$. |
| `report.md` | Comprehensive written technical report analyzing experimental methodology, PSNR/SSIM results, and computational trade-offs. |
| `test_images_greyscale/` | Input grayscale `.gif` benchmark images. |
| `results_greyscale/` | Generated denoised images, PSNR/SSIM tables (`psnrs.xlsx`, `ssims.xlsx`), and parameter sweep visualizations. |

---

## Recommended Parameters

<!-- IMPORTANT PARAMETER GUIDELINES:
- patchRadius: 3 (patch size = 6x6 or 7x7)
- searchWindowRadius: 9 (search region = 19x19)
- N_threshold: 40-70 patches (balances low-rank redundancy vs dissimilar patch corruption)
- c: 0.005 (balances singular value thresholding strength)
- delta: 0.1 (fraction of noise residual re-injected per outer loop)
- K: 3 (outer iterations; 3 is the sweet spot between PSNR gains and execution time)
-->

* **Patch radius (`patchRadius`)**: 3 (patch dimension $(2 \times 3)^2 = 36$ pixels)
* **Search window radius (`searchWindowRadius`)**: 9 ($19 \times 19$ search neighborhood)
* **Similar patch count (`N_threshold`)**: 50
* **Reweighting constant (`c`)**: 0.005
* **Regularization feedback factor (`delta`)**: 0.1
* **Outer iterations (`K`)**: 3

---

## Dependencies & Requirements

* Python 3.8+
* `numpy`
* `scipy`
* `scikit-image`
* `matplotlib`
* `pandas`
* `openpyxl` (for reading/writing `.xlsx` metric files)
