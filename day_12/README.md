# Day 12 — Keypoints : Harris Corner Detector

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Harris détecte les coins d'une image — les zones où l'intensité
change fortement dans toutes les directions. Un coin est le meilleur
type de keypoint car il est localisable sans ambiguïté, contrairement
à une zone plate (aucun changement) ou un bord (changement dans
une seule direction).

---

## 🧠 Key Ideas

- **Matrice de structure M** — construite à partir des gradients
  Ix et Iy. Ses valeurs propres λ1 et λ2 révèlent la nature
  de chaque zone : plate, bord ou coin
- **Score R** — évite de calculer les valeurs propres explicitement :
  `R = det(M) - k * trace(M)²`
  R >> 0 = coin / R << 0 = bord / |R| ≈ 0 = zone plate
- **Paramètre k** — contrôle la sélectivité (0.04 à 0.06)
- **threshold_ratio** — fraction du score R max utilisée comme seuil.
  Petit = plus de coins / Grand = moins de coins, plus sélectif
- **Précision sous-pixel** — `cornerSubPix` affine la position
  des coins au-delà du pixel entier. Indispensable en calibration
  caméra, reconstruction 3D et suivi d'objets

---

## 🐍 Script

`day12_harris.py` — chargement, calcul du score R via
`cv.cornerHarris`, seuillage, dilatation pour la visibilité,
affinage sous-pixel avec `cv.cornerSubPix`.
Comparaison visuelle : coins en rouge (pixel) vs vert (sous-pixel).

---

## 📐 Key Formula

Score de Harris :

$$
R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2
$$

Matrice de structure :

$$
M = \sum_{(x,y) \in W} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

---

## ⚠️ Limitations

- **Pas invariant à l'échelle** — si l'image est zoomée, les coins
  détectés changent. C'est la limitation principale qui a motivé
  SIFT (Day 14)
- **Sensible au bruit** — un flou gaussien préalable améliore
  les résultats sur les images bruitées
- **Pas de description** — Harris détecte les coins mais ne les
  décrit pas. Pour les matcher entre deux images, il faut ajouter
  un descripteur (SIFT, ORB...)