# Project Execution Guide

This guide explains how to set up, navigate, and run the grayscale image-denoising experiments in this repository. It is based on the repository's `README.md`, `report.md`, Python modules, and notebook source code.

## 1. Project overview

The main project implements **Weighted Nuclear Norm Minimization (WNNM)** for grayscale image denoising and compares it with three other denoising methods:

- Gaussian filtering
- Bilateral filtering
- Non-Local Means (NLM)

The notebooks load clean grayscale test images, add synthetic Gaussian noise, apply the denoising methods, and evaluate output quality using **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)**. The parameter-tuning notebooks use PSNR to compare WNNM parameter combinations.

The WNNM method groups similar image patches, applies singular-value decomposition (SVD) and weighted singular-value shrinkage, and combines overlapping patch estimates to form the denoised image.

## 2. Repository structure

| File or directory | Purpose |
|---|---|
| `README.md` | Project introduction, theory overview, recommended parameters, and dependency list. |
| `wnnm_final.py` | Main WNNM denoising function, `wnnm(...)`. |
| `filters_hw2.py` | Supporting Gaussian-kernel helper, bilateral filter, and non-local means filter. |
| `Results_greyscale_one_img.ipynb` | Compares Gaussian, bilateral, NLM, and WNNM on `12.gif`. |
| `Results_greyscale_all.ipynb` | Runs the four denoising methods on a set of 15 grayscale images and exports PSNR/SSIM spreadsheets. |
| `Parameter_tuning_c_n.ipynb` | Compares combinations of `c` and `N_threshold` using `2.gif`. |
| `Parameter_tuning_K_and_delta.ipynb` | Compares combinations of `K` and `delta` using `2.gif`. |
| `WNNM_Reproduction_2.ipynb` | A separate reproduction-and-extension workflow described in Section 7. |
| `test_images_greyscale/` | Input grayscale GIF images used by the main experiment notebooks. |
| `results_greyscale/` | Output location for denoised images, parameter-tuning images, and metric spreadsheets. |
| `report.md` | Technical report describing the project, experiments, results, and limitations. |

## 3. Requirements

The project README specifies **Python 3.8 or later**. The main grayscale notebooks and modules use:

- `numpy`
- `scipy`
- `scikit-image`
- `matplotlib`
- `pandas`
- `openpyxl` (used when exporting the Excel metric files)
- Jupyter Notebook or another Jupyter-compatible environment to run `.ipynb` files

The separate `WNNM_Reproduction_2.ipynb` notebook additionally imports **Pillow** (`PIL`). Installing Pillow is also useful if you intend to run that notebook.

The repository does not currently include a `requirements.txt` or a pinned package-version file. The commands below install the packages imported by the notebooks; they do not claim to reproduce a previously recorded, version-pinned environment.

## 4. Environment setup (Windows PowerShell)

Run these commands in a terminal opened at the repository's root directory (the folder containing `README.md`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-image matplotlib pandas openpyxl jupyter ipykernel pillow
```

If PowerShell blocks environment activation, you can use the Python executable inside the virtual environment directly, or follow your institution's permitted PowerShell execution-policy guidance. Do not change system-wide security settings just for this project.

In VS Code:

1. Open the repository root folder.
2. Open a notebook (`.ipynb`).
3. Select the Python interpreter/kernel from the `.venv` environment.
4. Confirm that the selected kernel is the same environment where the packages above were installed.

You can also start Jupyter from the repository root with:

```powershell
python -m jupyter notebook
```

## 5. Prepare the output folders

The main experiment notebooks save images and spreadsheets into subfolders under `results_greyscale/`. The notebook cells use these paths directly and do not create the folders themselves. Check that the following folders exist before running the corresponding notebooks:

- `results_greyscale/one_image/`
- `results_greyscale/clean/`
- `results_greyscale/noisy/`
- `results_greyscale/gaussian/`
- `results_greyscale/bilateral/`
- `results_greyscale/nlm/`
- `results_greyscale/wnnm/`
- `results_greyscale/wnnm_parameter_tuning/c_and_n/`
- `results_greyscale/wnnm_parameter_tuning/K_and_delta/`

To create any missing folders in Windows PowerShell, run this from the repository root:

```powershell
$folders = @(
    "results_greyscale/one_image",
    "results_greyscale/clean",
    "results_greyscale/noisy",
    "results_greyscale/gaussian",
    "results_greyscale/bilateral",
    "results_greyscale/nlm",
    "results_greyscale/wnnm",
    "results_greyscale/wnnm_parameter_tuning/c_and_n",
    "results_greyscale/wnnm_parameter_tuning/K_and_delta"
)
New-Item -ItemType Directory -Force -Path $folders | Out-Null
```

The command is safe to run even when some or all of the folders already exist.

## 6. Running the main grayscale experiments

**Important:** Run notebooks with the repository root as the working directory. Their input and output paths are relative to that location. Do not run individual cells from a different working directory unless you update the paths accordingly.

### 6.1 Single-image comparison

Open `Results_greyscale_one_img.ipynb` and run its cells from top to bottom. It:

1. Loads `test_images_greyscale/12.gif` and scales pixel values to the range used by the notebook.
2. Adds synthetic Gaussian noise with standard deviation `0.1`.
3. Applies Gaussian, bilateral, and NLM filtering.
4. Applies WNNM with `patchRadius = 3`, `delta = 0.1`, `c = 0.005`, `K = 3`, `sigma_n = 0.1`, and `N_threshold = 50`.
5. Calculates PSNR and SSIM and displays a visual comparison.
6. Saves the clean, noisy, and denoised images under `results_greyscale/one_image/`.

This is the recommended first notebook because it demonstrates the complete comparison on one image.

### 6.2 Parameter tuning for `c` and `N_threshold`

Open `Parameter_tuning_c_n.ipynb` and run its cells from top to bottom. It uses `test_images_greyscale/2.gif` and evaluates:

- `N_threshold`: `20`, `50`, and `100`
- `c`: `0.01`, `0.005`, and `0.001`

It computes PSNR for the combinations, displays the comparison, and saves images under `results_greyscale/wnnm_parameter_tuning/c_and_n/`.

### 6.3 Parameter tuning for `K` and `delta`

Open `Parameter_tuning_K_and_delta.ipynb` and run its cells from top to bottom. It uses `test_images_greyscale/2.gif` and evaluates:

- `K`: `1`, `3`, and `5`
- `delta`: `0.05`, `0.1`, and `0.25`

It computes PSNR for the combinations, displays the comparison, and saves images under `results_greyscale/wnnm_parameter_tuning/K_and_delta/`.

### 6.4 All-image benchmark

Open `Results_greyscale_all.ipynb` and run its cells from top to bottom. It evaluates 15 grayscale images (`2`, `7`, `12`, `13`, `15`, `16`, `20`, `22`, `23`, `25`, `27`, `28`, `38`, `47`, and `48`) using Gaussian, bilateral, NLM, and WNNM filtering. It calculates per-image PSNR and SSIM and exports:

- `results_greyscale/psnrs.xlsx`
- `results_greyscale/ssims.xlsx`

This notebook is the aggregate benchmark and may take longer to run than the single-image demonstration.

The report's suggested order is: install dependencies, run the single-image notebook, run the two parameter-tuning notebooks, and then run the all-image benchmark.

## 7. Separate notebook: `WNNM_Reproduction_2.ipynb`

`WNNM_Reproduction_2.ipynb` is a separate reproduction-and-extension workflow. It contains its own WNNM implementation rather than importing the main repository's `wnnm_final.py` function. Its sections cover a paper reproduction on the House image, algorithm checks, grayscale modifications, colour-image processing, and noise-estimation studies.

**Additional input required:** its first code cell expects a file named `house.png` in the same directory as the notebook. The notebook's own instructions say to obtain this image from the MATLAB repository. The file was not present in the project's top-level file listing when this guide was prepared, so this notebook will not run from a fresh copy of the listed project files until `house.png` is supplied in the required location.

The notebook sets `QUICK = False` by default and states that a full run can take roughly 20–30 minutes. It offers `QUICK = True` for a smaller smoke test. These are estimates stated in the notebook; actual run time depends on the computer. Its code also uses Pillow, in addition to NumPy, SciPy, scikit-image, pandas, and Matplotlib.

This notebook is not included in the main grayscale execution sequence above because it has a separate input prerequisite and workflow.

## 8. Understanding the evaluation metrics

- **PSNR (Peak Signal-to-Noise Ratio):** A measure of reconstruction error relative to the clean reference image. For the same image and compatible data range, a higher PSNR indicates lower reconstruction error.
- **SSIM (Structural Similarity Index):** A measure that compares structural similarity between the reference and denoised images. In the main grayscale notebooks, it is calculated using scikit-image.

The main experiment notebooks generate random Gaussian noise without setting a fixed random seed. As a result, repeated runs can produce different noisy images and different metric values. Do not expect every run to reproduce exactly the same PSNR/SSIM numbers.

## 9. Troubleshooting

| Problem | What to check |
|---|---|
| `ModuleNotFoundError` for a package | Confirm that the notebook is using the `.venv` kernel and install the missing package in that environment. |
| `ModuleNotFoundError` for `wnnm_final` or `filters_hw2` | Confirm that the notebook is running with the repository root as its working directory and that both `.py` files are present there. |
| `FileNotFoundError` for a GIF image | Confirm that the notebook is running from the repository root and that the named image exists in `test_images_greyscale/`. |
| Error while saving an image | Confirm that the required output subfolder under `results_greyscale/` exists; see Section 5. |
| `WNNM_Reproduction_2.ipynb` reports that `house.png` is missing | Place the required `house.png` next to that notebook, as described in Section 7. |

## 10. Notes on reproducibility

- The main notebooks use relative file paths, so the repository root should be the working directory.
- The main notebooks generate random noise without a fixed seed, so results may vary between runs.
- The WNNM and classical-filter implementations use nested loops and can be computationally expensive, especially for the all-image benchmark and parameter sweeps.
- The repository does not provide pinned dependency versions or an automated test suite. This guide documents the packages imported by the supplied files; it does not certify that every notebook has been run successfully on every system.
