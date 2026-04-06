# Day 06 — Denoising : Gaussian, Bilateral & Median

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Le bruit est inévitable en traitement d'images — capteur imparfait, compression,
transmission. Avant tout pipeline de vision, il faut le réduire sans écraser
les détails importants. Trois filtres classiques, trois comportements très différents.

---

## 🧠 Key Ideas

- **Bruit gaussien** — modèle de bruit le plus courant, simulé ici avec
  `np.random.normal` sur une image normalisée [0, 1]
- **Flou gaussien** — lisse uniformément, rapide, mais floute aussi les contours
- **Filtre bilatéral** — pondère les voisins par distance spatiale ET
  similarité d'intensité : lisse le bruit, préserve les bords
- **Filtre médian** — remplace chaque pixel par la médiane de son voisinage,
  redoutable contre le bruit sel & poivre, insensible aux valeurs extrêmes

---

## 🐍 Script

`day06_denoising.py` — pipeline complet : chargement, niveaux de gris,
normalisation, ajout de bruit gaussien synthétique, application des 3 filtres,
comparaison visuelle en grille 2×3.

---

## 📐 Key Formula

Filtre bilatéral — poids d'un pixel voisin $q$ pour lisser $p$ :

$$
w(p,q) = \exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{|I(p)-I(q)|^2}{2\sigma_r^2}\right)
$$

$\sigma_s$ = sensibilité spatiale · $\sigma_r$ = sensibilité aux intensités

---

## ⚠️ Limitations

- Le flou gaussien dégrade les contours — à éviter si la netteté est critique
- Le filtre bilatéral est lent sur les grandes images
- Le filtre médian peut effacer les fins détails si `ksize` est trop grand
- Aucun de ces filtres ne gère bien un bruit très intense
```