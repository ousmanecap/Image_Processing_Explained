<div align="center">

# 🖼️ From Pixel To Vision
### Classical Image Processing — A Structured Learning Series

*by **Ousmane Capinto CAMARA***

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![YouTube](https://img.shields.io/badge/YouTube-Series-FF0000?style=flat-square&logo=youtube&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🎯 What Is This Project?

**From Pixel To Vision** is a structured 20-day series exploring the fundamentals of classical image processing — from raw pixel manipulation to high-level scene understanding.

Each day combines three things:
- 🐍 **A Python script** — clean, commented code explaining *why*, not just *what*
- 🎥 **A YouTube video** — screen recording with slide-based walkthrough
- 📄 **A README** — concept summary, key formula, and video link for that day

> Built for two audiences: **beginners** discovering image processing for the first time, and **intermediate Python developers** looking to solidify their fundamentals.

---

## 📂 Repository Structure
```
from-pixel-to-vision/
│
├── day01/
│   ├── day01_histograms.py
│   ├── thumbnail.png
│   └── README.md
│
├── day02/
│   ├── day02_otsu_thresholding.py
│   ├── thumbnail.png
│   └── README.md
│
├── ...
│
├── day20/
│   ├── day20_fourier_transform.py
│   ├── thumbnail.png
│   └── README.md
│
├── utils/
│   ├── display.py           # Shared visualization helpers
│   └── metrics.py           # Quality metrics (PSNR, SSIM…)
│
├── data/
│   ├── natural/             # Photographs
│   ├── medical/             # Medical imaging
│   ├── satellite/           # Remote sensing
│   └── synthetic/           # Controlled test images
│
├── requirements.txt
└── README.md
```

---

## 📅 Series Overview — 20 Days

### 🗂️ Module 1 — Intensity & Histogram Analysis
| Day | Script | Topic |
|-----|--------|-------|
| 01 | `day01_histograms.py` | **Histograms & Equalization** — CDF, global EQ, CLAHE, LAB ✅ |
| 02 | `day02_otsu_thresholding.py` | **Otsu Thresholding** — bimodal histograms, between-class variance |
| 03 | `day03_adaptive_thresholding.py` | **Adaptive Thresholding** — local statistics, block strategies |

### 🗂️ Module 2 — Spatial Filtering & Morphology
| Day | Script | Topic |
|-----|--------|-------|
| 04 | `day04_smoothing_filters.py` | **Smoothing Filters** — box filter, Gaussian blur, convolution |
| 05 | `day05_edge_preserving.py` | **Edge-Preserving Filters** — bilateral, guided filter |
| 06 | `day06_morphology.py` | **Morphological Operations** — erosion, dilation, opening, closing |

### 🗂️ Module 3 — Denoising
| Day | Script | Topic |
|-----|--------|-------|
| 07 | `day07_gaussian_median_denoise.py` | **Gaussian & Median Denoising** — noise models, salt-and-pepper |
| 08 | `day08_bilateral_denoise.py` | **Bilateral Denoising** — spatial vs. range kernels |
| 09 | `day09_nlmeans.py` | **Non-Local Means** — patch similarity, weight computation |

### 🗂️ Module 4 — Segmentation
| Day | Script | Topic |
|-----|--------|-------|
| 10 | `day10_otsu_segmentation.py` | **Otsu Segmentation** — automatic thresholding, variance |
| 11 | `day11_watershed.py` | **Watershed** — topographic model, marker-controlled segmentation |
| 12 | `day12_graph_cut.py` | **Graph-Cut** — energy minimization, min-cut / max-flow |

### 🗂️ Module 5 — Keypoint Detection & Description
| Day | Script | Topic |
|-----|--------|-------|
| 13 | `day13_harris.py` | **Harris Corner Detector** — second-moment matrix, corner response |
| 14 | `day14_sift.py` | **SIFT** — scale-space, DoG, orientation & descriptor |
| 15 | `day15_orb.py` | **ORB** — FAST + BRIEF, binary descriptors, rotation invariance |

### 🗂️ Module 6 — Geometry & 3D Reconstruction
| Day | Script | Topic |
|-----|--------|-------|
| 16 | `day16_camera_calibration.py` | **Camera Model & Calibration** — pinhole model, distortion |
| 17 | `day17_stereo_vision.py` | **Stereo Vision & Disparity** — epipolar geometry, depth map |
| 18 | `day18_optical_flow.py` | **Optical Flow** — Lucas-Kanade, Farneback, motion estimation |
| 19 | `day19_reconstruction.py` | **2D/3D Reconstruction** — structure from motion, point cloud |

### 🗂️ Module 7 — Transforms & Compression
| Day | Script | Topic |
|-----|--------|-------|
| 20 | `day20_fourier_wavelet.py` | **Fourier & Wavelet Transforms** — DFT, frequency filtering, JPEG |

---

## 🛠️ Stack

| Library | Role |
|---------|------|
| **Python 3.10+** | Core language |
| **OpenCV** | Image I/O, filtering, geometric transforms |
| **NumPy** | Pixel-level array operations |
| **Matplotlib** | Visualization & plotting |
| **scikit-image** | Additional algorithms & metrics |
```bash
pip install -r requirements.txt
```

---

## 🧠 Learning Philosophy

> *"Understanding the math is as important as running the code."*

Every script follows the same structure:
1. **Imports & image loading** — real public domain image, no synthetic data
2. **Core algorithm** — step-by-step implementation with *why* comments
3. **Visualization** — side-by-side comparisons using Matplotlib
4. **Recap** — key takeaways summarized at the end of the script

Each day's `README.md` adds the theoretical context: historical background, mathematical foundation, video link, and limitations.

---

## 📊 Progress

| Module | Days | Status |
|--------|------|--------|
| Module 1 — Intensity & Histograms | 01 → 03 | 🟡 1/3 done |
| Module 2 — Spatial Filtering | 04 → 06 | ⬜ Not started |
| Module 3 — Denoising | 07 → 09 | ⬜ Not started |
| Module 4 — Segmentation | 10 → 12 | ⬜ Not started |
| Module 5 — Keypoint Detection | 13 → 15 | ⬜ Not started |
| Module 6 — Geometry & 3D | 16 → 19 | ⬜ Not started |
| Module 7 — Transforms | 20 | ⬜ Not started |

---

## 🎥 YouTube Series

Each day has a dedicated video — screen recording + slides walkthrough.  
The link is available in each day's `README.md`.

👉 **[Ousmane Capinto CAMARA — YouTube](#)**

---

<div align="center">

*From a single pixel to a complete scene — one script at a time.*

</div>
