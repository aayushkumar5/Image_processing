"""
Classical Image Denoising Filters (Gaussian, Bilateral, and Non-Local Means).
Adapted and extended from Stanford EE367: Computational Imaging (Winter 2022).

This module provides baseline spatial and non-local filtering implementations
to compare against modern low-rank matrix minimization techniques like WNNM:
1. 2D Gaussian Filter (Isotropic spatial smoothing)
2. Bilateral Filter (Range + Spatial edge-preserving smoothing)
3. Non-Local Means (NLM) Filter (Patch-similarity based weighted averaging)
"""

import numpy as np
from skimage.filters import gaussian

# Adapted from HW2 of EE367: Computational imaging winter quarter 2022

def fspecial_gaussian_2d(size, sigma):
    """
    Generate a 2D isotropic Gaussian convolution kernel normalized to unit sum.

    Mathematical Formulation:
        G(u, v) = (1 / (2 * pi * sigma^2)) * exp(-(u^2 + v^2) / (2 * sigma^2))
        Kernel is normalized such that sum_{u, v} G(u, v) = 1.0 to preserve DC energy.

    Parameters:
        size (tuple/list of int): Spatial dimensions (height, width) of the kernel window.
        sigma (float): Standard deviation of the Gaussian spatial distribution.

    Returns:
        numpy.ndarray: Normalized 2D Gaussian kernel of shape (size[0], size[1]).
    """
    kernel = np.zeros(tuple(size))
    # Place a Dirac delta impulse at the kernel center
    kernel[size[0]//2, size[1]//2] = 1
    # Filter impulse with continuous Gaussian and normalize
    kernel = gaussian(kernel, sigma)
    return kernel / np.sum(kernel)

def bilateral2d(img, radius, sigma, sigmaIntensity):
    """
    2D Bilateral Filter for edge-preserving image denoising (Tomasi & Manduchi, 1998).

    Mathematical Formulation:
        Combines a spatial Gaussian domain filter with a photometric radiometric range filter:
            I_filtered(p) = (1 / W_p) * sum_{q in Omega} I(q) * G_sigma(||p - q||) * G_sigmaIntensity(|I(p) - I(q)|)
        where normalization factor W_p is:
            W_p = sum_{q in Omega} G_sigma(||p - q||) * G_sigmaIntensity(|I(p) - I(q)|)

    Important Notes:
        - Geometric closeness: Penalizes pixels distant in spatial coordinates.
        - Radiometric similarity: Penalizes pixels distant in intensity values.
        - Edges are preserved because across high-contrast boundaries, radiometric weights approach zero.

    Parameters:
        img (numpy.ndarray): Input 2D grayscale image array of shape (H, W).
        radius (int): Half-window size; neighbourhood window is (2*radius + 1, 2*radius + 1).
        sigma (float): Spatial domain Gaussian standard deviation (spatial distance decay).
        sigmaIntensity (float): Range domain Gaussian standard deviation (intensity difference decay).

    Returns:
        numpy.ndarray: Bilateral filtered grayscale image with preserved edge structures.
    """
    pad = radius
    # Initialize filtered output image buffer to 0
    out = np.zeros_like(img)

    # Replicate/symmetric padding to minimize boundary artifacts at image borders
    imgPad = np.pad(img, pad)

    # Precompute spatial domain smoothing kernel G_sigma of size (2*radius+1, 2*radius+1)
    filtSize = (2*radius + 1, 2*radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    
    # Iterate across each pixel in the interior image grid
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Center pixel intensity value in padded coordinate system
            centerVal = imgPad[y+pad, x+pad] # Careful of padding amount!

            # Compute photometric range weights based on intensity differences:
            # intensityKernel = exp(- (I(q) - I(p))^2 / (2 * sigmaIntensity^2))
            neighborhood = imgPad[y+pad-radius:y+pad+radius+1, x+pad-radius:x+pad+radius+1]
            intensityKernel = np.exp(-((neighborhood - centerVal)**2) / (2 * sigmaIntensity**2))

            # Combine spatial and radiometric kernels for joint bilateral weight
            jointKernel = intensityKernel * spatialKernel
            Wp = np.sum(jointKernel)
            
            # Normalize by sum of weights W_p to preserve local photometric equilibrium
            out[y, x] = (1.0 / Wp) * np.sum(neighborhood * jointKernel)
    return out

def nonlocalmeans(img, searchWindowRadius, averageFilterRadius, sigma, nlmSigma):
    """
    Non-Local Means (NLM) Filter for texture and structure preservation (Buades et al., 2005).

    Mathematical Formulation:
        Unlike local filters that assume spatial locality, NLM exploits non-local self-similarity:
            NL[u](x) = sum_{y in SearchWindow} w(x, y) * u(y)
        where weights w(x, y) measure the L2 Euclidean distance between comparison patches:
            w(x, y) = (1 / Z(x)) * exp(- ||u(N_x) - u(N_y)||_{2, a}^2 / (2 * nlmSigma^2))
        where ||.||_{2, a} is a Gaussian-weighted patch norm using spatial standard deviation sigma.

    Important Implementation Details:
        - Search Window: Region of radius searchWindowRadius searched for candidate patches.
        - Patch Window: Local comparison patch of size (2*averageFilterRadius + 1)^2.
        - Self-weight handling: w(x, x) is replaced by max_{y != x} w(x, y) to prevent
          the central pixel from excessively dominating the weighted sum under heavy noise.

    Parameters:
        img (numpy.ndarray): Input 3D image array of shape (H, W, C).
        searchWindowRadius (int): Radius of the non-local search neighbourhood window.
        averageFilterRadius (int): Radius of patch used for computing similarity metrics.
        sigma (float): Spatial standard deviation for Gaussian weighting inside patches.
        nlmSigma (float): Filtering parameter h controlling decay rate of similarity weights.

    Returns:
        numpy.ndarray: Non-locally filtered image of shape (H, W, C).
    """
    # Initialize output to 0
    out = np.zeros_like(img)
    # Total padding required for combined search window and patch comparison radius
    pad = averageFilterRadius + searchWindowRadius
    imgPad = np.pad(img, pad)
    imgPad = imgPad[..., pad:-pad] # Don't pad third channel (color/channel dimension)

    # Compute Gaussian spatial weighting kernel for within-patch distance computation
    filtSize = (2*averageFilterRadius + 1, 2*averageFilterRadius + 1)
    kernel = fspecial_gaussian_2d(filtSize, sigma)
    # Add third axis for proper array broadcasting across channels
    kernel = kernel[:, :, np.newaxis]
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Extract central reference patch N_x
            centerPatch = imgPad[y+pad-averageFilterRadius:y+pad+averageFilterRadius+1,
                                 x+pad-averageFilterRadius:x+pad+averageFilterRadius+1,
                                 :]
            # Extract full candidate search window around current reference pixel
            searchWindowImage = imgPad[y+pad-searchWindowRadius:y+pad+searchWindowRadius+1,
                                 x+pad-searchWindowRadius:x+pad+searchWindowRadius+1,
                                 :]
            weights = np.zeros((2*searchWindowRadius+1, 2*searchWindowRadius+1, 1))
            
            # Efficiently compute patch distances using vertically sliced sliding windows
            for i in range(2*searchWindowRadius+1):
                otherPatch = imgPad[y+pad-searchWindowRadius-averageFilterRadius:y+pad+searchWindowRadius+averageFilterRadius+1,
                                    x+pad-searchWindowRadius+i-averageFilterRadius:x+pad-searchWindowRadius+i+averageFilterRadius+1,
                                    :]
                
                # Compute linear indexer for sliding patch extractions
                indexer = np.arange((2*averageFilterRadius+1)**2)[None, :] + (2*averageFilterRadius+1)*np.arange(otherPatch.shape[0]-2*averageFilterRadius)[:, None]
                
                otherPatch = otherPatch.flatten()
                otherPatch = np.reshape(otherPatch[indexer], (otherPatch[indexer].shape[0], 2*averageFilterRadius + 1, 2*averageFilterRadius + 1, 1))
                
                # Compute Gaussian-weighted Euclidean L2 distance between patches and convert to exponential weights
                weights[:, i] = np.exp(-np.sum(kernel * (centerPatch - otherPatch)**2, axis=(1, 2)) / (2 * nlmSigma**2))
            
            # Prevent central pixel self-weight singularity from dominating under noise:
            # Set center weight to zero temporarily, then assign the maximum non-central weight
            weights[searchWindowRadius, searchWindowRadius] = 0
            weights[searchWindowRadius, searchWindowRadius] = np.max(weights)
            
            # Normalize weights to sum to 1 across the search window
            weights = weights / np.sum(weights)
            
            # Compute final weighted non-local average for pixel (y, x)
            out[y, x, :] = np.sum(weights * searchWindowImage)
    return out
