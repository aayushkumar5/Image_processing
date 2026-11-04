Test images used in WNNM_Reproduction.ipynb

house.png (256x256, grey) comes from the MATLAB repo and is used full size in Part C (Table I).
All other experiments use the 128x128 crops in crops_128x128_used_in_experiments/.

crop            source image (scikit-image)       rows      cols      used in
house_crop      House (from repo), grey            64:192    64:192    Tables II, VII, VIII, appendix G
camera_crop     skimage.data.camera(), grey        100:228   150:278   Tables II, VII, VIII, appendix G
coins_crop      skimage.data.coins(), grey         60:188    100:228   Table II
astronaut_crop  skimage.data.astronaut(), colour   40:168    150:278   Tables III, V, VI
chelsea_crop    skimage.data.chelsea(), colour     60:188    150:278   Tables III, V, VI, Fig. 2
coffee_crop     skimage.data.coffee(), colour      100:228   200:328   Tables V, VI, Fig. 3 and the real-type-noise test (Table IV uses the full House image)

The full source images are in full_images/ for reference. These are the clean images; the noisy inputs are
generated inside the notebook (fixed seeds) by adding Gaussian or signal-dependent noise.
