# Day 01 — Histograms & Equalization

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

A histogram is the fingerprint of an image — it tells you how pixel intensities
are distributed across the 0–255 range. A dark image clusters values near 0,
an overexposed one near 255, a well-contrasted image spreads them evenly.

Histogram equalization redistributes those intensities to maximize contrast
by using the Cumulative Distribution Function (CDF) as a remapping curve.
CLAHE (Contrast Limited Adaptive Histogram Equalization) goes further by
working on local tiles instead of the full image — preventing over-amplification
of noise in uniform regions.

---

## 🧠 Key Ideas

- A grayscale image is a 2D NumPy array of shape `(H, W)` with values in [0, 255]
- The histogram counts how many pixels have each intensity value
- The CDF tells you what fraction of pixels fall *below* a given intensity
- Global equalization stretches the CDF to be as linear as possible
- CLAHE limits the amplification per tile via `clipLimit` to avoid artifacts
- For color images, equalization is applied on the **L channel** (LAB space),
  not on RGB directly — to avoid color distortion

---

## 🐍 Script

`day01_histograms.py` — loads a real grayscale image, computes histogram and
CDF, applies global equalization and CLAHE, then compares results visually.
Also demonstrates color equalization via the LAB color space.

---

## 📐 Key Formula

Global equalization remaps each intensity $v$ using the CDF:

$$
v' = \text{round}\left( \frac{CDF(v) - CDF_{min}}{(H \times W) - CDF_{min}} \times 255 \right)
$$

Where:
- $CDF(v)$ = number of pixels with intensity ≤ $v$
- $CDF_{min}$ = first non-zero value of the CDF
- $H \times W$ = total number of pixels

---

## ⚠️ Limitations

- **Global equalization** can over-amplify noise in low-contrast regions
- **CLAHE** solves this but introduces a `clipLimit` hyperparameter to tune
- Neither method works well on already well-exposed images — the result
  can look unnatural
- Applying equalization directly on RGB channels distorts colors —
  always use the L channel in LAB space for color images