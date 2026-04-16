# Day 11 — Segmentation : Graph-Cut

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Graph-Cut modélise l'image comme un graphe où chaque pixel est un nœud.
Deux nœuds spéciaux — Source (objet) et Puits (fond) — sont reliés à
chaque pixel via des arêtes pondérées. L'algorithme trouve la coupe
minimale qui sépare les deux : c'est le Min-Cut, équivalent au Max-Flow.

L'énergie minimisée combine deux termes :
- **Terme région** — probabilité qu'un pixel ressemble à l'objet ou au fond
- **Terme frontière** — coût de séparer deux pixels voisins similaires

---

## 🧠 Key Ideas

- **Graines (scribbles)** — l'utilisateur dessine des traits sur l'objet
  et le fond. Ces échantillons construisent les modèles de couleur GMM
- **GMM** — deux mélanges gaussiens modélisent la couleur typique de
  l'objet et du fond depuis les graines
- **Min-Cut = Max-Flow** — théorème fondamental qui garantit une
  solution optimale globale, pas locale
- **GC_INIT_WITH_MASK** — mode masque plus précis que le rectangle
  du Day 09 car on contrôle exactement les pixels objet et fond
- **Raffinement** — fermeture + ouverture morphologique + lissage
  gaussien pour corriger les défauts du masque brut

---

## 🐍 Script

`day11_graphcut.py` — interface souris interactive pour dessiner les
graines, Graph-Cut via `cv.grabCut` en mode masque, raffinement
morphologique, visualisation des 4 labels et de l'objet extrait.

---

## 📐 Key Formula

Énergie globale à minimiser :

$$
E(\alpha) = \sum_{p} R_p(\alpha_p) + \lambda \sum_{(p,q) \in \mathcal{N}} B_{pq} \cdot \mathbb{1}[\alpha_p \neq \alpha_q]
$$

Terme frontière entre deux pixels voisins $p$ et $q$ :

$$
B_{pq} = \exp\left(-\frac{\|I(p) - I(q)\|^2}{2\sigma^2}\right)
$$

---

## ⚠️ Limitations

- Sensible à la qualité des graines — dessiner sur les bords donne
  de mauvais résultats
- Objets avec couleurs similaires au fond = segmentation imprécise
- Plus lent que Otsu et Watershed sur les grandes images
- Ne segmente qu'un seul objet par appel