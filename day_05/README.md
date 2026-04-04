# Day 05 — Hough Transform : Lines & Circles

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

La transformée de Hough permet de détecter des formes géométriques simples
(droites, cercles) dans une image — même si elles sont partiellement cachées
ou bruitées. L'idée : chaque pixel de contour "vote" pour toutes les formes
qui passent par lui. Là où les votes s'accumulent, il y a une forme.

---

## 🧠 Key Ideas

- **Prérequis** — Hough travaille sur des pixels de contours, pas sur l'image
  brute. Un passage par Canny est indispensable
- **HoughLines** — détecte des droites infinies en espace polaire (ρ, θ).
  Chaque pixel vote pour toutes les droites possibles
- **HoughLinesP** — version probabiliste : tire aléatoirement des sous-ensembles
  de pixels, retourne directement des segments (x1, y1, x2, y2), plus rapide
- **HoughCircles** — détecte des cercles, applique Canny en interne.
  Paramètres `param1` et `param2` à bien calibrer pour éviter les faux positifs

---

## 🐍 Script

`day05_hough_transform.py` — pipeline complet avec 4 fonctions modulaires :
extraction des contours, détection de lignes standard, probabiliste, et cercles.
Comparaison visuelle des 4 résultats sur une même figure.

---

## 📐 Key Formula

Représentation polaire d'une droite :

$$
\rho = x \cdot \cos\theta + y \cdot \sin\theta
$$

Chaque pixel $(x, y)$ vote pour toutes les paires $(\rho, \theta)$ qui
vérifient cette équation — le pic dans l'accumulateur révèle la droite.

---

## ⚠️ Limitations

- Sensible au bruit — un bon pré-traitement Canny est indispensable
- `HoughLines` retourne des droites infinies — pas toujours exploitable
- `HoughCircles` ne détecte qu'un seul rayon à la fois et rate les cercles
  partiels
- Les paramètres (threshold, minDist, param2) demandent du tuning manuel
```

---