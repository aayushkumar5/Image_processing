"""
Weighted Nuclear Norm Minimization (WNNM) for Image Denoising.

Reference:
    Shuhang Gu, Lei Zhang, Wangmeng Zuo, and Xiangchu Feng.
    "Weighted Nuclear Norm Minimization with Application to Image Denoising."
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 2862-2869.
    DOI: 10.1109/CVPR.2014.366

Theoretical Overview:
    Standard Nuclear Norm Minimization (NNM) assumes all singular values are penalized
    equally by a constant threshold lambda. However, singular values of natural image patch
    groups possess distinct physical meanings:
      - Large singular values represent major structural energy (edges, contours, flat backgrounds).
      - Smaller singular values primarily encode high-frequency noise and residual artifacts.
    WNNM generalizes NNM by assigning non-descending weights (w_1 <= w_2 <= ... <= w_r)
    to singular values, penalizing larger singular values less and penalizing smaller ones more:
        min_X  ||Y - X||_F^2 + ||X||_{w, *}
    where ||X||_{w, *} = sum_i (w_i * sigma_i(X)).

Algorithm Key Steps:
    1. Iterative Regularization: Adds back a fraction delta of residual noise (Y - X_hat)
       to avoid loss of faint textures across iterations.
    2. Non-Local Patch Grouping: For each local reference patch, finds N_threshold closest
       patches within a search window using Euclidean distance to form matrix Y_j.
    3. Zero-Mean Centering: Subtracts the mean patch so low-rank matrix estimation models
       structural variance rather than DC illumination.
    4. Closed-form SVD Thresholding: Computes SVD Y_j = U * Sigma * V^T, estimates clean
       singular values via variance subtraction, updates adaptive weights w_i = c * sqrt(m) / (sigma_i + eps),
       and applies generalized singular value thresholding (SVT).
    5. Overlap Patch Aggregation: Reconstructed patches are accumulated into the full image canvas
       and normalized by an overlap counter matrix.
"""

import numpy as np
from scipy.linalg import svd

def wnnm(img, patchRadius, delta, c, K, sigma_n, N_threshold):
    """
    Apply Weighted Nuclear Norm Minimization (WNNM) denoising to an input image.

    Parameters:
        img (numpy.ndarray): Noisy input image of shape (H, W, C) with pixel intensities in [0, 1].
        patchRadius (int): Half-size of the square comparison patch. Full patch is (2*patchRadius, 2*patchRadius).
        delta (float): Relaxation parameter for iterative residual feedback regularization (typical: 0.1).
        c (float): Positive hyperparameter controlling the strength of adaptive singular value weights (typical: 0.005).
        K (int): Number of outer iterations for iterative regularization and re-estimation (typical: 3 - 5).
        sigma_n (float): Standard deviation of Gaussian noise contaminating the input image.
        N_threshold (int): Number of similar patches grouped together into matrix Y_j (typical: 40 - 70).

    Returns:
        numpy.ndarray: Denoised image of shape (H, W, C) with intensities clipped to [0, 1].
    """
    # Define spatial search window radius (3x patch size provides good balance of diversity & speed)
    searchWindowRadius = patchRadius * 3
    
    # Internal iterations to iteratively solve non-convex reweighted singular values
    N_iter = 3
    
    # Compute border padding width to handle boundary pixels without out-of-bounds indexing
    pad = searchWindowRadius + patchRadius
    imgPad = np.pad(img, pad_width=pad)
    imgPad = imgPad[..., pad:-pad] # Retain color/third channel unpadded
    
    # Initialize the iterative estimate xhat_iter with the input noisy image
    xhat_iter = img
    
    # Outer iterative regularization loop (K iterations)
    for n in range(K):
        # Pad current estimate to match search geometry
        xhat_iter = np.pad(xhat_iter, pad_width=pad)
        xhat_iter = xhat_iter[..., pad:-pad]
        
        # Iterative regularization: inject fraction delta of residual noise back to prevent over-smoothing:
        # y_iter = xhat^{(k)} + delta * (y - xhat^{(k)})
        y_iter = xhat_iter + delta * (imgPad - xhat_iter)
        
        # Matrix tracking how many overlapping patches contribute to each pixel (for weighted averaging)
        pixel_contribution_matrix = np.ones_like(imgPad)
        
        # Iterate over all pixel positions as patch centers
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                # 1. Extract central reference patch (size: 2*patchRadius x 2*patchRadius)
                centerPatch = y_iter[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                     i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                     :]
                
                # Pre-allocate distance vector and patch candidate buffer
                num_search_pixels = (2 * searchWindowRadius + 1) ** 2
                patch_dim = (2 * patchRadius) ** 2
                dists = np.ones((num_search_pixels))
                patches = np.zeros((num_search_pixels, patch_dim))
                
                # 2. Vectorized patch extraction across vertical slices in search window
                for k in range(2 * searchWindowRadius + 1):
                    # Vertical slice covering all candidate patches along column offset k
                    otherPatch = y_iter[j:j+2*pad,
                                        i+k:i+k+2*patchRadius,
                                        :]
                    
                    # 1D sliding window indexing arithmetic to extract all patches in column slice
                    indexer = np.arange((2*patchRadius)**2)[None, :] + (2*patchRadius) * np.arange(otherPatch.shape[0]-2*patchRadius+1)[:, None]
                    
                    otherPatch = otherPatch.flatten()
                    otherPatch = np.reshape(otherPatch[indexer], (otherPatch[indexer].shape[0], (2*patchRadius)**2))
                    
                    # Compute Mean Squared Error (MSE) Euclidean distance to reference patch
                    start_idx = k * (2 * searchWindowRadius + 1)
                    end_idx = (k + 1) * (2 * searchWindowRadius + 1)
                    dists[start_idx:end_idx] = (np.sum((centerPatch.reshape(((2*patchRadius)**2)) - otherPatch)**2, axis=1) / (2*patchRadius)**2).flatten()
                    patches[start_idx:end_idx, :] = otherPatch
                    
                # 3. Non-Local Grouping: Select top N_threshold nearest patches and construct patch matrix Yj
                indcs = np.argsort(dists)
                Yj = (patches[indcs[:N_threshold], :]).transpose() # Shape: (patch_dim, N_threshold)
                
                # 4. Zero-Mean Centering: Remove patch DC component
                Yj_means = np.sum(Yj, axis=0)
                Yj_center = Yj - Yj_means
                
                # 5. SVD Decomposition: Yj_center = U * S * V^T
                U, S, V_T = svd(Yj_center, full_matrices=False)
                
                # Initial estimate of clean singular values by subtracting expected noise variance:
                # sigma_i(X) = sqrt(max(S_i^2 - m * sigma_n^2, 0))
                sing_val = np.sqrt(np.maximum(S**2 - N_threshold * (sigma_n**2), 0))
                
                # 6. Iterative reweighting for Non-Convex WNNM:
                # Adaptive weights: w_i = c * sqrt(m) / (sigma_i(X) + eps)
                # Larger singular values get smaller weights (preserve structure),
                # smaller singular values get larger weights (suppress noise).
                for m in range(N_iter):
                    w = c * np.sqrt(N_threshold) / (sing_val + 1e-6)
                    sing_val = np.diag(np.maximum(S - w, 0))
                
                # Reconstruct low-rank estimate: Xj_hat = U * max(S - w, 0) * V^T + Yj_means
                Xj_hat_center = U @ np.diag(np.maximum(S - w, 0)) @ V_T
                Xj_hat = Xj_hat_center + Yj_means
                
                # 7. Accumulate estimated denoised central patch (first column of Xj_hat) into canvas
                denoised_patch = np.clip(Xj_hat[:, 0].reshape((2*patchRadius, 2*patchRadius, 1)), 0, 1)
                xhat_iter[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                          i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                          :] = xhat_iter[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                         i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                         :] + denoised_patch
                
                # Update contribution counter for sliding window overlap averaging
                pixel_contribution_matrix[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                          i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                          :] = pixel_contribution_matrix[j+searchWindowRadius:j+searchWindowRadius+2*patchRadius,
                                                                         i+searchWindowRadius:i+searchWindowRadius+2*patchRadius,
                                                                         :] + np.ones_like(denoised_patch)
        
        # Crop borders and normalize each pixel by the number of patch contributions
        xhat_iter = xhat_iter[pad:-pad, pad:-pad, :] / pixel_contribution_matrix[pad:-pad, pad:-pad, :]
        
    # Return final reconstructed denoised image array
    out = xhat_iter
    return out
