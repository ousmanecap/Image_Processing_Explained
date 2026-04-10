# Day 10 — Segmentation : Watershed Algorithm

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Watershed traite une image comme un relief topographique : les pixels
sombres sont des vallées, les pixels clairs des collines. On "remplit
d'eau" ce relief depuis des marqueurs placés dans les vallées. Là où
deux bassins se rejoignent = frontière entre deux objets.
Contrairement à Otsu, Watershed segmente plusieurs objets
simultanément, même s'ils se touchent.

---

## 🧠 Key Ideas

- **Binarisation Otsu inversée** — objets = blanc, fond = noir.
  Convention indispensable pour les étapes suivantes
- **Ouverture morphologique** — supprime le bruit avant de construire
  les marqueurs, sinon fausses frontières garanties
- **Fond certain** — dilatation des objets : ce qui reste en dehors
  = fond certain à 100%
- **Distance Transform** — chaque pixel blanc reçoit sa distance au
  fond le plus proche. Le centre d'un objet = valeur maximale
- **Zone inconnue** = fond certain - objet certain. C'est là que
  Watershed décide
- **Marqueurs +1** — décalage obligatoire pour que Watershed distingue
  le fond (label 1) de la zone inconnue (label 0)
- **Frontières = -1** — après Watershed, les pixels de frontière
  reçoivent la valeur -1, colorés en rouge pour visualisation

---

## 🐍 Script

`day10_watershed.py` — pipeline complet en 8 fonctions modulaires :
chargement, binarisation, nettoyage, fond certain, Distance Transform,
zone inconnue, marqueurs, Watershed. Comparaison visuelle en grille 2×4.

---

## 📐 Key Formula

Distance Transform — valeur de chaque pixel blanc $p$ :

$$
D(p) = \min_{q \in \text{fond}} \| p - q \|_2
$$

Seuillage pour obtenir l'objet certain :

$$
\text{sure\_fg}(p) = \begin{cases} 255 & \text{si } D(p) > \alpha \cdot D_{max} \\ 0 & \text{sinon} \end{cases}
$$

$\alpha$ = `threshold_ratio` — contrôle la taille de la zone certaine.

---

## ⚠️ Limitations

- Sensible au bruit — le nettoyage morphologique préalable est
  indispensable
- Sur-segmentation possible si les marqueurs sont trop nombreux
  ou mal placés
- `threshold_ratio` à calibrer selon l'image — trop bas fusionne
  des objets, trop haut en rate certains
- Fonctionne mieux sur des objets de taille et forme similaires