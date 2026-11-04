# EE367 WNNM Denoising Project Report

<!-- 
CRITICAL PROJECT TAKEAWAY & EXECUTIVE SUMMARY:
- Weighted Nuclear Norm Minimization (WNNM) provides superior edge and texture reconstruction
  compared to local filters (Gaussian, Bilateral) and patch-averaging filters (NLM).
- Unlike nuclear norm minimization (NNM) which applies equal soft-thresholding to all singular values,
  WNNM dynamically assigns weights inversely proportional to the singular value magnitude (w_i = c*sqrt(m)/(sigma_i + eps)).
- Optimal parameters empirically determined for Gaussian noise sigma = 0.1:
  patchRadius = 3, N_threshold = 50, c = 0.005, K = 3, delta = 0.1.
- Benchmarked across 15 standard images using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity).
-->

## 1. Project Purpose

This repository is an academic implementation of Weighted Nuclear Norm Minimization (WNNM) for grayscale image denoising. It compares WNNM with Gaussian filtering, bilateral filtering, and non-local means filtering.

The experiments use clean grayscale images, add synthetic Gaussian noise, apply the denoising methods, and evaluate the results using PSNR and SSIM.

The repository is primarily notebook-driven. The Python files provide the reusable filter implementations, while the notebooks load images, select parameters, run experiments, save images, and calculate metrics.

PDF files are intentionally excluded from this report.

## 2. Repository Structure

| Path | Role |
| --- | --- |
| `README.md` | Project introduction and notebook/file overview. |
| `wnnm_final.py` | Main WNNM denoising implementation. |
| `filters_hw2.py` | Bilateral and non-local means filter implementations, plus a Gaussian kernel helper. |
| `Results_greyscale_one_img.ipynb` | Runs all denoising methods on one image. |
| `Results_greyscale_all.ipynb` | Runs all methods on a collection of grayscale images and writes aggregate metrics. |
| `Parameter_tuning_c_n.ipynb` | Evaluates different WNNM values of `c` and `N_threshold`. |
| `Parameter_tuning_K_and_delta.ipynb` | Evaluates different WNNM values of `K` and `delta`. |
| `test_images_greyscale/` | Input grayscale `.gif` test images. |
| `results_greyscale/` | Generated denoised images, tuning outputs, and metric spreadsheets. |

The repository also contains Git metadata and a generated `__pycache__/` directory. These are infrastructure/runtime artifacts rather than project logic.

## 3. File-by-File Description

### `README.md`

This is the project documentation. It identifies the project as an implementation of the WNNM image-denoising paper and explains the purpose of the Python modules and notebooks.

It documents:

- The WNNM implementation file.
- The classical comparison filters.
- The one-image experiment.
- The all-image experiment.
- The two parameter-tuning experiments.
- The input and output directories.

It does not currently document installation commands, Python package versions, expected image dimensions, runtime estimates, or a reproducible execution procedure.

### `wnnm_final.py`

This file implements the `wnnm` function:

```python
wnnm(img, patchRadius, delta, c, K, sigma_n, N_threshold)
```

#### Implemented behavior

1. Sets the search-window radius to three times `patchRadius`.
2. Pads the input image so patches can be extracted near image boundaries.
3. Performs `K` outer denoising iterations.
4. Creates a regularized image using the current estimate and the original noisy image.
5. Iterates over every image pixel.
6. Extracts a central patch and all candidate patches in the local search window.
7. Computes squared patch distances.
8. Selects the `N_threshold` most similar patches.
9. Forms a patch matrix whose columns are the selected patches.
10. Applies SVD using SciPy.
11. Estimates noise-adjusted singular values.
12. Iteratively computes weights and shrinks singular values.
13. Reconstructs the denoised central patch.
14. Adds patch estimates into an accumulation image and averages overlapping contributions.
15. Returns the final denoised image.

#### Input and output expectations

The notebooks pass grayscale images with shape `(height, width, 1)`. The returned array has the same three-dimensional shape. Pixel values are expected to be normalized approximately to `[0, 1]`.

#### Important implementation considerations

- The algorithm is computationally expensive because it combines image-sized loops, search-window loops, patch sorting, and repeated SVD operations.
- `Yj_means` is calculated with `np.sum(Yj, axis=0)`. Despite its name, this is not a mean. A conventional column-centering operation would use `np.mean(Yj, axis=0)` or divide the sum by the number of patch pixels.
- The function has no explicit validation for dimensionality, positive parameters, or `N_threshold` being within the number of available candidate patches.
- The implementation clips each reconstructed central patch before accumulating it.

### `filters_hw2.py`

This module contains classical denoising filters adapted from an EE367 homework implementation.

#### `fspecial_gaussian_2d(size, sigma)`

Creates a 2D impulse image, applies a Gaussian filter, and normalizes the result so the kernel sums to one. The helper is used by the bilateral and non-local means implementations.

#### `bilateral2d(img, radius, sigma, sigmaIntensity)`

Implements bilateral filtering with:

- A spatial Gaussian kernel.
- An intensity-based Gaussian weight.
- Explicit zero padding at image boundaries.
- Per-pixel weighted averaging.

It is intended for 2D grayscale floating-point arrays.

#### `nonlocalmeans(img, searchWindowRadius, averageFilterRadius, sigma, nlmSigma)`

Implements a patch-based non-local means filter. For every pixel it:

1. Extracts a local reference patch.
2. Extracts candidate patches from a search window.
3. Computes Gaussian similarity weights from patch differences.
4. Replaces the center weight with the maximum weight from other patches.
5. Produces a weighted average of the search-window pixels.

The notebooks add a singleton channel dimension before calling this function and remove it afterward.

#### Module limitations

- The module depends on `scikit-image` but the repository has no dependency file.
- The filters use explicit Python loops and may be slow for larger images.
- The output uses `np.zeros_like(img)`, so integer input arrays can cause unintended integer truncation.
- `max_weight` is assigned in `nonlocalmeans` but is not subsequently used.

### `Results_greyscale_one_img.ipynb`

This notebook is the single-image demonstration and comparison workflow.

#### Implemented behavior

- Loads `test_images_greyscale/12.gif`.
- Converts the image to floating point and scales it by 255.
- Adds Gaussian noise with standard deviation `0.1`.
- Runs Gaussian filtering.
- Runs bilateral filtering.
- Runs non-local means.
- Runs WNNM.
- Computes PSNR and SSIM for the noisy and filtered images.
- Displays the clean, noisy, and denoised images.
- Saves the comparison images under `results_greyscale/one_image/`.

This notebook is useful for visually comparing the methods on one fixed image, but the noise is randomly generated without a seed, so repeated executions can produce different metrics.

### `Results_greyscale_all.ipynb`

This notebook evaluates the filters on a group of 15 named grayscale images.

#### Implemented behavior

- Loads the listed `.gif` images from `test_images_greyscale/`.
- Generates a noisy version of each image.
- Saves clean and noisy images.
- Applies Gaussian filtering to every image.
- Applies bilateral filtering to every image.
- Applies non-local means to every image.
- Applies WNNM to every image.
- Computes per-image PSNR and SSIM values.
- Writes `psnrs.xlsx` and `ssims.xlsx` into `results_greyscale/`.

The notebook is the main aggregate experiment. It relies on the current working directory being the repository root and on the output subdirectories already existing.

### `Parameter_tuning_c_n.ipynb`

This notebook studies two WNNM parameters while keeping the other parameters fixed.

It evaluates:

- `N_thresholds = [20, 50, 100]`.
- `cs = [0.01, 0.005, 0.001]`.

For every parameter combination, it runs WNNM on `test_images_greyscale/2.gif`, computes PSNR, displays the results in a grid, and saves the generated images under `results_greyscale/wnnm_parameter_tuning/c_and_n/`.

### `Parameter_tuning_K_and_delta.ipynb`

This notebook studies the number of outer WNNM iterations and the regularization parameter.

It evaluates:

- `Ks = [1, 3, 5]`.
- `deltas = [0.05, 0.1, 0.25]`.

For every combination, it runs WNNM on `test_images_greyscale/2.gif`, computes PSNR, displays the results in a grid, and saves the generated images under `results_greyscale/wnnm_parameter_tuning/K_and_delta/`.

## 4. Input Data Directory

### `test_images_greyscale/`

This directory contains the grayscale `.gif` images used by the notebooks. The notebooks load these files with `skimage.io.imread`, convert the arrays to floating point, and divide by 255.

The data is used in two ways:

- Image `12.gif` for the one-image comparison.
- Image `2.gif` for parameter tuning.
- A list of 15 image names for the all-image comparison.

## 5. Results Directory

### `results_greyscale/`

This directory stores generated experiment artifacts, including:

- Clean images.
- Noisy images.
- Gaussian-filtered images.
- Bilateral-filtered images.
- Non-local-means images.
- WNNM images.
- One-image comparison results.
- WNNM parameter-tuning results.
- `psnrs.xlsx` containing PSNR measurements.
- `ssims.xlsx` containing SSIM measurements.

These files are outputs rather than independent implementation modules.

## 6. Dependencies

The code imports or uses:

- Python 3.
- NumPy.
- SciPy.
- scikit-image.
- pandas.
- Matplotlib.
- Jupyter/IPython.
- An Excel writer supported by pandas, typically `openpyxl`.

The current environment successfully compiles the two Python files, but importing `filters_hw2.py` currently fails because `scikit-image` is not installed.

## 7. Main Limitations and Risks

1. There is no `requirements.txt` or environment file.
2. There is no command-line entry point.
3. There are no automated tests.
4. Experiments depend on relative paths and notebook working-directory state.
5. Random noise generation is not seeded.
6. WNNM and the classical filters use slow nested Python loops.
7. WNNM assumes a three-dimensional image representation.
8. WNNM uses a sum where a patch mean appears intended for centering.
9. Several notebook imports and variables are unused, reflecting exploratory coursework code.
10. The notebooks mix experiment logic, plotting, file output, and metric calculation in large cells, making reuse and debugging harder.

## 8. Recommended Execution Order

1. Install the required Python packages.
2. Open Jupyter from the repository root.
3. Run `Results_greyscale_one_img.ipynb` first to confirm the environment.
4. Run `Parameter_tuning_c_n.ipynb` and `Parameter_tuning_K_and_delta.ipynb` to inspect parameter sensitivity.
5. Run `Results_greyscale_all.ipynb` for the aggregate comparison.

Before relying on the reported WNNM metrics, verify the intended patch-centering calculation in `wnnm_final.py` and record a fixed random seed for reproducibility.