# Color Calibration Script

This script estimates a **Color Correction Matrix (CCM)** from a Macbeth ColorChecker and applies it to an image. It can optionally perform gamma correction and histogram equalization.

---

## Features

* Samples 24 ColorChecker patches (hard‑coded coordinates).
* Computes a least‑squares CCM mapping camera sRGB → calibrated sRGB.
* Optional gamma handling and histogram equalization.
* Saves:

  * `image_data.csv` – measured patch means.
  * `final_result.png` – corrected image.
  * `comparison.png` – side‑by‑side input vs. output.

---

## Usage

1. Install dependencies:

   ```bash
   pip install numpy opencv-python matplotlib pillow colour-science
   ```

2. Place your input image in `input_images/` and set the path in `calibrate.py`.

3. Run:

   ```bash
   python calibrate.py
   ```

4. Outputs will appear in the project folder.

---

## Notes

* Patch coordinates in `calculate_patch_means` must match your image’s chart.
---

## Credits

* [`colour-science`](https://www.colour-science.org/) for reference data.
* OpenCV and Matplotlib for I/O and visualization.
