# Red Eye Correction with OpenCV

This project is a C++ image processing tool that automatically detects and corrects the red-eye effect in digital photographs. The system uses OpenCV and implements custom preprocessing, morphological filtering, and color analysis in the HSV space to accurately locate and fix affected eye regions.

---

## Overview

The red-eye effect occurs when a camera flash reflects off the retina, producing a bright red glow in the pupils. This project solves the problem in two main stages:

1. **Eye Region Detection** (Preprocessing)
2. **Red Eye Correction** (HSV Filtering & Pixel Modification)

---

## Features

- Convert color images to grayscale and binary masks
- Morphological operations (opening) to reduce noise
- Connected component labeling with geometric filtering:
  - Area
  - Aspect ratio
  - Thinness ratio
  - Vertical position (eyes are centered on the face)
- HSV-based red pixel detection
- Red pixel correction via blending (not simple desaturation)

---

