# Term Project: Image Super Resolution

Team 5 — 110612117 張仲瑜 (Chung-Yu Chang), 110550128 蔡曜霆, 110550109 陳芳靖

This project explores **single-image super-resolution (SR)** with a focus on practical image quality improvements through model choice and lightweight pre/post-processing. Our experiments were driven primarily by **visual quality** (edge sharpness, texture recovery, and reduced artifacts) rather than only chasing benchmark numbers.

The full write-up and figures are in `Team_5.pdf`.

## Method Overview

### Base Model

We follow **SwinIR**:

- Paper: *SwinIR: Image Restoration Using Swin Transformer* (2108.10257)
- Key idea (as used in our project):
  - Shallow features are extracted first.
  - Deep features are modeled with **Residual Swin Transformer Blocks (RSTB)**.
  - The final SR image is produced by combining deep and shallow features.

In our runs, we mainly used **SwinIR-Large** as the SR backbone.

## Experiments & Improvements

### A. Pre-processing Experiments (Before SR)

We tested several pre-processing techniques to help the SR model recover more detail from low-resolution inputs:

1. **No pre-processing**: directly apply SwinIR-Large.
2. **Denoising**: OpenCV non-local means (e.g., `cv2.fastNlMeansDenoisingColored`).
3. **Sharpening kernel**: spatial sharpening to emphasize edges.
4. **Histogram equalization**: increase contrast to amplify local differences.
5. **Multi-scale detail enhancement**: edge-preserving smoothing + detail boosting.

**Observation:** after comparing results, we ultimately **did not adopt any pre-processing step**. Although pre-processing could make fine structures more visible, it tended to **hurt overall visual quality** or introduce artifacts after SR.

### B. Post-processing Experiments (After SR)

We evaluated post-processing strategies aimed at improving perceptual quality after SR:

1. **No post-processing**: directly use SwinIR-Large output.
2. **Unsharp masking** (Pillow `ImageFilter.UnsharpMask`): enhances edges and fine details, but can also amplify noise.
3. **Gaussian blur** (Pillow): reduces graininess / harsh artifacts caused by aggressive sharpening.

**Observation:** because noise becomes relatively small and localized after SR, applying **unsharp masking in the post-processing stage** produced better-looking results than using it earlier. In some cases, a light Gaussian blur helped suppress over-sharpening artifacts.

### C. Cascaded Upscaling + Downscaling (Model Chaining)

To further improve visual sharpness, we tested a “**super-resolution then downscale**” strategy:

1. **No chaining**: single SR pass.
2. **SR ×16, then bicubic downscale ×4**.
3. **SR ×64, then bicubic downscale ×16**.
4. **Two-stage chaining + unsharp masking**: cascade SR, then downscale, then sharpen (parameters tuned).

**Observation:** the best perceived quality came from **two-stage chaining plus unsharp masking**, followed by downscaling. This approach often preserved more local details than a single-pass ×4 SR.

## Results

The report shows qualitative comparisons on multiple images (e.g., cat, car, landscape, dog, dessert, flower, portrait, and other samples). The final pipeline produces sharper edges and clearer textures while keeping artifacts acceptable.

## Reflections

- We tried multiple SR papers and implementations; many did not perform as expected on our inputs.
- We initially aimed to upsample directly to ×4, but model size/parameter constraints affected performance.
- A practical workaround was **cascaded SR** (e.g., ×2 twice) paired with **downscaling** to the target resolution.
- For downscaling, we tested Gaussian blur (anti-aliasing) + bicubic interpolation; however, too much smoothing removed fine detail.
- We found that **unsharp masking combined with bicubic interpolation** achieved a strong balance in visual quality.

