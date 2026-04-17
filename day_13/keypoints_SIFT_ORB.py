# =============================================================
# DAY 13 — KEYPOINTS : SIFT & ORB
# =============================================================
# CONCEPTS CLÉS À RETENIR :
#
# 1. SIFT (Scale Invariant Feature Transform) :
#    - Détecte les keypoints dans un espace-échelle (DoG)
#    - Attribue une orientation dominante → invariance rotation
#    - Décrit chaque keypoint par un vecteur de 128 valeurs
#    - Robuste mais lent
#
# 2. ORB (Oriented FAST + Rotated BRIEF) :
#    - FAST détecte les keypoints en comparant les voisins
#    - BRIEF décrit par des comparaisons binaires (256 bits)
#    - Beaucoup plus rapide que SIFT
#    - Libre de droits
#
# 3. KEYPOINT — ce qu'il contient :
#    .pt       = coordonnées (x, y)
#    .size     = échelle de détection
#    .angle    = orientation dominante
#    .response = score de force du keypoint
#    .octave   = niveau de la pyramide gaussienne
#
# 4. DIFFÉRENCE FONDAMENTALE :
#    Harris → détecte uniquement (pas de description)
#    SIFT   → détecte + décrit (128 float32 par keypoint)
#    ORB    → détecte + décrit (256 bits par keypoint)
# =============================================================

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# On remonte deux niveaux depuis le script pour trouver la racine du projet
# Ensuite on construit le chemin complet vers l'image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'kobi.png')


# =========================
# 1. Chargement
# =========================

def load_image(path):
    """
    Charge l'image depuis le disque en format BGR.
    SIFT et ORB travaillent en niveaux de gris mais on garde
    l'image couleur pour pouvoir dessiner les keypoints dessus
    en couleur et les voir clairement.
    """
    img = cv.imread(path)
    # Si l'image n'existe pas ou le chemin est faux, on arrête tout
    # avec un message d'erreur clair plutôt que de continuer en silence
    assert img is not None, f"Image non chargée : {path}"
    return img

def to_gray(img_bgr):
    """
    Convertit l'image couleur BGR en niveaux de gris.
    SIFT et ORB calculent des gradients d'intensité —
    un seul canal suffit, pas besoin des 3 canaux couleur.
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def to_rgb(img_bgr):
    """
    Convertit BGR en RGB avant d'afficher avec Matplotlib.
    Sans cette conversion, le rouge et le bleu sont inversés
    à l'affichage — les keypoints verts apparaîtraient en orange.
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


# =========================
# 2. Détection SIFT
# =========================

def detect_sift(img_gray, n_features=500):
    """
    Détecte et décrit les keypoints avec SIFT.

    SIFT construit une pyramide gaussienne de l'image à plusieurs
    niveaux de zoom et cherche les extrema du DoG (Difference of
    Gaussians) — les points qui ressortent à la fois dans l'espace
    et dans l'échelle.

    Pour chaque keypoint trouvé, SIFT calcule :
    → une orientation dominante depuis l'histogramme de gradients
      locaux — c'est ce qui donne l'invariance à la rotation
    → un descripteur de 128 valeurs construit sur une grille 4x4
      avec 8 orientations par cellule (4 x 4 x 8 = 128)

    n_features → nombre maximum de keypoints à retourner.
                 0 = pas de limite, l'algo retourne tout ce qu'il trouve.

    detectAndCompute fait les deux étapes en un seul appel :
    détection des keypoints ET calcul de leurs descripteurs.

    Retourne :
    keypoints   → liste d'objets cv.KeyPoint
    descriptors → tableau numpy de forme (N, 128) en float32
                  chaque ligne = le descripteur d'un keypoint
    """
    # On crée le détecteur SIFT avec le nombre de keypoints souhaité
    sift = cv.SIFT_create(nfeatures=n_features)

    # On détecte les keypoints ET on calcule leurs descripteurs en un coup
    # None = pas de masque, on traite toute l'image
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    print(f"[SIFT] {len(keypoints)} keypoints détectés")
    return keypoints, descriptors


# =========================
# 3. Détection ORB
# =========================

def detect_orb(img_gray, n_features=500):
    """
    Détecte et décrit les keypoints avec ORB.

    FAST parcourt l'image pixel par pixel et teste chaque pixel
    contre ses 16 voisins disposés sur un cercle de rayon 3.
    Si au moins 9 voisins consécutifs sont tous plus clairs ou
    tous plus sombres que le pixel central → c'est un keypoint.
    C'est très rapide car c'est juste des comparaisons d'intensité.

    BRIEF calcule le descripteur en comparant des paires de pixels
    tirées aléatoirement autour du keypoint :
    si pixel_a > pixel_b → on note 1, sinon → on note 0
    On répète 256 fois → on obtient un vecteur de 256 bits.

    ORB oriente FAST et BRIEF selon l'orientation locale du patch
    autour de chaque keypoint → invariance à la rotation.

    n_features → nombre maximum de keypoints à retourner.

    Retourne :
    keypoints   → liste d'objets cv.KeyPoint
    descriptors → tableau numpy de forme (N, 32) en uint8
                  32 octets = 256 bits, un bit par comparaison
    """
    # On crée le détecteur ORB avec le nombre de keypoints souhaité
    orb = cv.ORB_create(nfeatures=n_features)

    # Même logique que SIFT — détection + description en un seul appel
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    print(f"[ORB] {len(keypoints)} keypoints détectés")
    return keypoints, descriptors


# =========================
# 4. Visualisation des keypoints
# =========================

def draw_keypoints(img_bgr, keypoints, color=(0, 255, 0)):
    """
    Dessine les keypoints sur une copie de l'image originale.

    cv.drawKeypoints parcourt la liste de keypoints et pour chacun :
    → dessine un cercle dont le rayon correspond à la taille du keypoint
      (l'échelle à laquelle il a été détecté)
    → dessine une ligne indiquant l'orientation dominante du keypoint

    DRAW_RICH_KEYPOINTS est le flag qui active l'affichage complet :
    taille ET orientation. Sans ce flag, tous les keypoints
    auraient le même petit cercle sans information d'échelle.

    color → couleur des cercles en BGR
            on passe du vert pour SIFT et de l'orange pour ORB
            pour les distinguer facilement à l'affichage.

    On travaille sur une copie pour ne jamais modifier l'image originale.
    """
    # img_bgr.copy() est fait en interne par drawKeypoints
    # None = on laisse OpenCV créer l'image de sortie
    img_kp = cv.drawKeypoints(
        img_bgr, keypoints, None,
        color=color,
        flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    return img_kp


# =========================
# 5. Visualisation grille
# =========================

def show_images(images, titles, cmap_list=None):
    """
    Affiche toutes les images dans une grille 2 x N.

    (n+1)//2 calcule le bon nombre de colonnes pour ranger
    n images sur 2 lignes — par exemple 3 images donnent 2 colonnes.
    zip() associe chaque image à son titre et sa cmap en une seule boucle.
    cmap=None laisse Matplotlib choisir automatiquement
    — il choisit RGB si l'image a 3 canaux.
    """
    n = len(images)
    cols = (n + 1) // 2
    rows = 2

    if cmap_list is None:
        cmap_list = ['gray'] * n

    plt.figure(figsize=(4 * cols, 8))

    for i, (img, title, cmap) in enumerate(zip(images, titles, cmap_list)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# =========================
# 6. Pipeline complet
# =========================

def main():
    # On charge l'image depuis le disque
    img_bgr  = load_image(img_path)
    # On convertit en niveaux de gris pour les algorithmes
    img_gray = to_gray(img_bgr)
    # On convertit en RGB uniquement pour l'affichage Matplotlib
    img_rgb  = to_rgb(img_bgr)

    # On détecte les keypoints SIFT et on les dessine en vert
    kp_sift, desc_sift = detect_sift(img_gray, n_features=500)
    img_sift     = draw_keypoints(img_bgr, kp_sift, color=(0, 255, 0))
    img_sift_rgb = to_rgb(img_sift)

    # On détecte les keypoints ORB et on les dessine en orange
    kp_orb, desc_orb = detect_orb(img_gray, n_features=500)
    img_orb     = draw_keypoints(img_bgr, kp_orb, color=(0, 165, 255))
    img_orb_rgb = to_rgb(img_orb)

    # On affiche dans le terminal la forme et le type des descripteurs
    # C'est important pour comprendre la différence entre SIFT et ORB :
    # SIFT → 128 valeurs float32 par keypoint (vecteur dense)
    # ORB  → 32 octets uint8 par keypoint (256 bits, vecteur binaire)
    print(f"\n[SIFT] Shape descripteur : {desc_sift.shape}")
    print(f"       Type             : {desc_sift.dtype}")
    print(f"[ORB]  Shape descripteur : {desc_orb.shape}")
    print(f"       Type             : {desc_orb.dtype}")

    # On prépare les 3 images à afficher côte à côte
    images = [
        img_rgb,
        img_sift_rgb,
        img_orb_rgb
    ]

    titles = [
        "Originale",
        f"SIFT — {len(kp_sift)} keypoints (vert)",
        f"ORB  — {len(kp_orb)} keypoints (orange)"
    ]

    # None = Matplotlib choisit la colormap (RGB pour les images couleur)
    cmap_list = [None, None, None]

    show_images(images, titles, cmap_list=cmap_list)


if __name__ == "__main__":
    main()