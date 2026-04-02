# Day 03 — Edge Detection : Sobel & Canny

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

La détection de contours consiste à repérer les zones où l'intensité des
pixels change brutalement — c'est là que se trouvent les bords des objets.
Deux approches classiques : Sobel qui mesure le gradient, Canny qui va plus
loin en affinant et filtrant les contours détectés.

---

## 🧠 Key Ideas

- **Flou Gaussien** — appliqué avant tout pour réduire le bruit, sinon
  chaque grain devient un faux contour
- **Sobel** — calcule le gradient de l'image sur l'axe X, Y ou les deux.
  Utiliser `CV_64F` + `np.absolute()` pour ne pas perdre les bords négatifs
- **Canny** — pipeline complet : flou → gradient → NMS → double seuillage.
  Plus précis et moins bruité que Sobel

---

## 🐍 Script

`day03_edge_detection.py` — applique un flou Gaussien, détecte les contours
avec Sobel et Canny, puis compare les deux résultats sur une image réelle.

---

## 📐 Key Formula

Gradient Sobel combiné :

$$
G = \sqrt{G_x^2 + G_y^2}
$$

Seuillage par hystérésis Canny :

$$
\text{si } G > t_{max} \Rightarrow \text{contour fort}
\quad
\text{si } G < t_{min} \Rightarrow \text{rejeté}
\quad
\text{sinon} \Rightarrow \text{contour faible (gardé si connecté)}
$$

---

## ⚠️ Limitations

- Sobel est sensible au bruit — toujours pré-filtrer
- Canny dépend fortement du choix de `threshold1` et `threshold2`
- Les deux peinent sur les images avec éclairage non uniforme