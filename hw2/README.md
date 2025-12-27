# Image Processing HW2 — Histogram Ops

Implements grayscale histogram equalization and histogram specification (matching) with NumPy/OpenCV/Matplotlib. The script loads the provided test images, shows intermediate histograms/CDFs, and writes the processed outputs alongside the originals.

## Files
- `110612117.py`: main script with equalization and specification routines.
- `Q1.jpg`: source for histogram equalization.
- `Q2_source.jpg`, `Q2_reference.jpg`: source and reference for histogram specification.
- `Q1_histogtam_equalized_img.jpg`: saved result for Q1.
- `Q2_histogram_specification.jpg`: saved result for Q2.

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
   cd hw/hw2
   ```
2. Run the script:
   ```bash
   python 110612117.py
   ```
3. Close the Matplotlib windows to continue. Outputs are written with the filenames above.

## Implementation notes
- Histogram equalization: builds the CDF, masks zeros to avoid division by zero, scales to `[0, 255]`, and remaps the original image.
- Histogram specification: computes CDFs for source and reference, finds nearest CDF matches per intensity to create a 256-level mapping, and applies it pixelwise.
- Histograms and CDF curves are displayed before/after to verify distribution changes.

## Customization
- Swap in different grayscale inputs by changing the filenames near the bottom of `110612117.py`.
- Comment out `show_histogram` calls to suppress interactive plots when running in batch mode.
