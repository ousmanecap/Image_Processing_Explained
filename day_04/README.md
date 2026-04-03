# Day 04 — Morphological Operations

![Thumbnail](thumbnail.png)

## 🎥 Video
👉 [Watch on YouTube](#)

---

## 📌 Concept

Les opérations morphologiques travaillent sur la **forme** des objets dans
une image binaire. En faisant glisser un élément structurant sur l'image,
on peut supprimer du bruit, combler des trous, ou extraire des contours —
sans toucher aux intensités.

---

## 🧠 Key Ideas

- **Érosion** — réduit les objets, supprime les petits éléments isolés
- **Dilatation** — agrandit les objets, comble les petits trous
- **Ouverture** (érosion → dilatation) — supprime le bruit de fond
- **Fermeture** (dilatation → érosion) — comble les trous à l'intérieur des objets
- **Gradient morphologique** — différence entre dilatation et érosion, donne les contours
- L'**élément structurant** (forme + taille) définit le comportement de chaque opération

---

## 🐍 Script

`day04_morphology.py` — applique seuillage adaptatif puis compare érosion,
dilatation, ouverture, fermeture et gradient morphologique sur une image réelle.

---

## 📐 Key Formula

$$
\text{Ouverture} = (A \ominus B) \oplus B
\quad
\text{Fermeture} = (A \oplus B) \ominus B
$$

$$
\text{Gradient} = (A \oplus B) - (A \ominus B)
$$

---

## ⚠️ Limitations

- Fonctionne uniquement sur des **images binaires** — le seuillage préalable est indispensable
- La taille du kernel impacte fortement le résultat — trop grand et on déforme les objets
- Ne gère pas bien les images avec éclairage non uniforme sans pré-traitement