# Day 13 — Keypoints : SIFT & ORB

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Harris détectait des coins mais sans les décrire ni les rendre
invariants à l'échelle. SIFT et ORB vont plus loin : ils détectent
ET décrivent chaque keypoint avec un vecteur numérique — ce qui
permet de les retrouver et de les matcher entre deux images
différentes de la même scène.

---

## 🧠 Key Ideas

- **SIFT** — construit une pyramide gaussienne et cherche les extrema
  du DoG à plusieurs échelles. Chaque keypoint reçoit une orientation
  dominante et un descripteur de 128 valeurs float32
- **ORB** — FAST détecte les keypoints par comparaisons d'intensité
  sur un cercle de 16 voisins. BRIEF les décrit par 256 comparaisons
  binaires → vecteur de 32 octets en uint8
- **DRAW_RICH_KEYPOINTS** — affiche la taille ET l'orientation de
  chaque keypoint — le rayon du cercle = l'échelle de détection
- **SIFT vs ORB sur une vraie image** — SIFT trouve les structures
  géométriques régulières (carrelage, bâtiments). ORB se concentre
  sur les zones à fort contraste local irrégulier (yeux, textures)
- **128 float32 vs 256 bits** — SIFT est plus riche mais plus lent.
  ORB est binaire → comparaison par XOR → 10 à 100x plus rapide

---

## 🐍 Script

`day13_sift_orb.py` — chargement, détection SIFT et ORB avec
`detectAndCompute`, dessin des keypoints avec `drawKeypoints`,
comparaison visuelle des deux détecteurs sur la même image.

---

## 📐 Key Formula

Descripteur SIFT — 128 valeurs construites sur une grille 4×4×8 :

$$
d_{SIFT} \in \mathbb{R}^{128} \quad (4 \times 4 \text{ cellules} \times 8 \text{ orientations})
$$

Descripteur ORB — 256 comparaisons binaires :

$$
d_{ORB}(p) = \sum_{i=1}^{256} 2^{i-1} \cdot \mathbb{1}[I(p_i) > I(q_i)]
$$

---

## ⚠️ Limitations

- **SIFT** — lent sur les grandes images, pas adapté au temps réel
- **ORB** — moins précis que SIFT, sensible aux grands changements
  d'échelle
- **Les deux** — ne fonctionnent bien que si les keypoints sont
  reproductibles entre les deux images à matcher
- **Harris vs SIFT/ORB** — Harris retourne des pixels bruts sans
  descripteur. SIFT/ORB retournent des keypoints structurés
  et descriptibles → indispensables pour le feature matching