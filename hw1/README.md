# Image Processing HW1 — Geometric Transforms

Implements image rotation and enlargement using three interpolation strategies (nearest neighbor, bilinear, bicubic) in pure NumPy/OpenCV. The code operates on the provided `building.jpg` and exports the processed results as JPEGs.

## Files
- `110612117.py`: main script with transform implementations and a simple preview pipeline.
- `building.jpg`: sample input image used by default.
- `Nearest Neighbor Enlarged Image.jpg`, `Biliear Enlarged Image.jpg`, `Bicubic Enlarged Image.jpg`: upscaled outputs.
- `Nearest_Neighbor Rotated Image.jpg`, `Bilinear Rotated Image.jpg`, `Bicubic Rotated Image.jpg`: rotated outputs (clockwise 30°).

## Environment
- Python 3.8+
- Dependencies: `numpy`, `opencv-python`

Install deps:
```bash
pip install numpy opencv-python
```

## How to run
1. Move to the homework folder:
   ```bash
   cd hw/hw1
   ```
2. Run the script:
   ```bash
   python 110612117.py
   ```
3. Close each preview window to proceed to the next. All outputs are saved automatically to the same directory with the filenames listed above.

## Implementation notes
- Rotation: manual backward mapping around the image center with nearest, bilinear, and bicubic sampling (clockwise 30° by default).
- Enlargement: manual backward mapping on a regular grid for 2× scaling, again with nearest, bilinear, and bicubic interpolation.
- Bicubic helper clamps channel values into `[0, 255]` to avoid overflow before casting to `uint8`.

## Customization
- Change the input file by editing `image = cv2.imread('building.jpg')` near the bottom of the script.
- Adjust the rotation angle by passing a different `angle` to the rotation functions.
- Adjust the scaling factor by passing a different `scale` to the enlargement functions.
- To run without GUI pop-ups, comment out the `show_image` calls at the bottom; the files will still be written.
