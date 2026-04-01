# Day 02 — Thresholding & Otsu Method

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Thresholding is the simplest form of image segmentation — it converts a
grayscale image into a binary image by deciding, for each pixel, whether
it belongs to the foreground or the background.

The challenge: choosing the right threshold. Too low, too high — and the
result falls apart. Otsu's method solves this automatically by finding the
threshold that maximizes the separation between two pixel classes.

---

## 🧠 Key Ideas

- **Global thresholding** — one fixed threshold for the entire image, works
  well only under uniform lighting
- **Adaptive thresholding** — threshold computed locally per region (Mean or
  Gaussian weighted), handles uneven lighting
- **Otsu's method** — finds the optimal threshold automatically by maximizing
  between-class variance on the histogram

---

## 🐍 Script

`day02_thresholding.py` — applies and compares global thresholding, adaptive
thresholding (Mean & Gaussian), and Otsu's method on a real grayscale image.

---

## 📐 Key Formula

Otsu maximizes the between-class variance $\sigma_B^2$:

$$
\sigma_B^2(t) = w_0(t) \cdot w_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2
$$

Where $w_0$, $w_1$ are the class probabilities and $\mu_0$, $\mu_1$ their
means — computed directly from the histogram.

---

## ⚠️ Limitations

- Global thresholding fails under non-uniform lighting
- Otsu assumes a **bimodal histogram** — it struggles with complex scenes
- Adaptive thresholding is sensitive to `blockSize` and `C` parameters