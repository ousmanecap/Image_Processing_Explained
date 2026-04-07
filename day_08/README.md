# Day 07 — NL-Means Denoising

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Les filtres classiques lissent le bruit en regardant uniquement les pixels
voisins. NL-Means (Non-Local Means) va plus loin : il cherche dans toute
l'image les patches similaires et fait une moyenne pondérée par ressemblance.
Un pixel loin mais dans un contexte visuel proche influence plus qu'un
voisin dans un contexte différent.

---

## 🧠 Key Ideas

- **Patch** — petite fenêtre de pixels autour du pixel à débruiter
- **Similarité** — deux patches sont similaires si leur distance euclidienne
  est faible
- **Moyenne non-locale** — chaque pixel est remplacé par la moyenne pondérée
  de tous les pixels dont le patch ressemble au sien
- **Paramètre `h`** — contrôle la force du débruitage. Trop grand = image
  trop lisse, trop petit = bruit résiduel
- `cv.fastNlMeansDenoising` — implémentation optimisée d'OpenCV pour les
  images en niveaux de gris

---

## 🐍 Script

`day07_nlmeans.py` — pipeline complet : chargement, niveaux de gris,
normalisation, bruit gaussien, puis comparaison de 4 filtres (gaussien,
bilatéral, médian, NL-Means) sur une grille 2×3.

---

## 📐 Key Formula

$$
NL[u](x) = \frac{1}{C(x)} \int e^{-\frac{(G_a * |u(x+.) - u(y+.)|^2)(0)}{h^2}} \, u(y) \, dy
$$

Chaque pixel $x$ est remplacé par une moyenne de tous les pixels $y$,
pondérée par la similarité de leurs patches. $h$ contrôle la décroissance
des poids.

---

## ⚠️ Limitations

- Plus lent que les filtres classiques — coût computationnel élevé
- Le paramètre `h` est sensible : trop grand efface les textures fines
- `searchWindowSize` grand = meilleure qualité mais encore plus lent
- Ne gère pas bien les images avec structures très répétitives