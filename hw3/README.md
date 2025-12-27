# Image Processing HW3 — Sharpening Filters

Implements Laplacian-based sharpening in both spatial and frequency domains for color images. The script loads `Q1.jpg`, applies two spatial kernels (cross and 8-neighbor variants), and a frequency-domain Laplacian to produce sharpened outputs; each result is shown and saved.

## Files
- `110612117.py`: main script with custom convolution, spatial kernels, and FFT-based Laplacian sharpening.
- `Q1.jpg`: input image.
- `output/`: saved images when the script runs (e.g., `Cross_Kernel Filtered Image.jpg`, `Around_Kernel Filtered Image.jpg`, `New Image.jpg`).

## Environment
- Python 3.8+
- Dependencies: `numpy`, `opencv-python`, `matplotlib`

Install deps:
```bash
pip install numpy opencv-python matplotlib
```

## How to run
1. Move to the homework folder:
   ```bash
   cd hw/hw3
   ```
2. Execute the script:
   ```bash
   python 110612117.py
   ```
3. Close each OpenCV window to proceed; results are also written as JPEGs in the same directory.

## Implementation notes
- Spatial filtering: custom convolution with flipped kernels; two Laplacian-like masks (cross and full 8-neighbor) highlight edges before adding back to the image.
- Frequency filtering: normalizes each channel, computes FFT, applies a centered Laplacian mask in the frequency domain, inverse-transforms, and performs unsharp masking (`new_channel = channel - k * laplacian`).
- Outputs are clipped to `[0, 255]` and cast to `uint8` to avoid overflow.

## Customization
- Adjust sharpening strength by changing `c` (for spatial kernels) and `k` (frequency-domain unsharp masking) in `110612117.py`.
- Swap in another input by editing the `cv2.imread('Q1.jpg', 1)` line.
- Comment out `show_image` calls to suppress GUI windows when running headless.
