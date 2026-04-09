# Day 09 — Segmentation : Otsu & GrabCut

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

La segmentation consiste à séparer les objets du fond dans une image.
Deux approches très différentes comparées ici :
Otsu segmente automatiquement par seuillage global, GrabCut extrait
un objet précis grâce à un rectangle fourni par l'utilisateur et des
modèles de couleur gaussiens (GMM).

---

## 🧠 Key Ideas

- **Otsu** — trouve automatiquement le seuil qui maximise la variance
  inter-classes. Suppose un histogramme bimodal. Résultat : image binaire
- **Otsu + flou** — le flou gaussien préalable lisse l'histogramme et
  rend les deux pics plus distincts → seuil plus robuste
- **GrabCut** — algorithme semi-automatique par graph cut. Modélise
  fond et objet avec des GMM couleur (BGR). L'utilisateur fournit
  un rectangle englobant l'objet via `selectROI`
- **4 labels GrabCut** — fond certain (0), objet certain (1),
  fond probable (2), objet probable (3)
- **bitwise_and** — applique le masque binaire sur l'image originale
  pour extraire l'objet sur fond noir

---

## 🐍 Script

`day09_segmentation_otsu_grabcut.py` — pipeline complet : chargement,
Otsu simple, Otsu + flou, sélection interactive du rectangle via
`selectROI`, GrabCut, comparaison visuelle en grille 2×3.

---

## 📐 Key Formula

Otsu — variance inter-classes à maximiser :

$$
\sigma_B^2(t) = w_0(t) \cdot w_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2
$$

GrabCut — énergie globale à minimiser :

$$
E = \lambda \cdot R(\alpha) + B(\alpha)
$$

$R$ = terme région (GMM) · $B$ = terme frontière · $\lambda$ = poids

---

## ⚠️ Limitations

- Otsu échoue si l'histogramme n'est pas bimodal
- GrabCut rate si fond et objet ont des couleurs similaires
- GrabCut segmente **un seul objet** par appel
- Le rectangle doit englober **entièrement** l'objet sinon
  les parties en dehors sont considérées comme fond définitivement