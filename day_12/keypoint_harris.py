# =============================================================
# DAY 12 — KEYPOINTS : Harris Corner Detector
# =============================================================
# CONCEPTS CLÉS À RETENIR :
#
# 1. QU'EST-CE QU'UN KEYPOINT ?
#    Un keypoint = un point de l'image qui est facilement
#    localisable et reproductible — même si l'image change
#    de taille, d'orientation ou d'éclairage.
#    Les coins sont les meilleurs keypoints car ils varient
#    dans TOUTES les directions.
#
# 2. MATRICE DE STRUCTURE M :
#    Construite à partir des gradients Ix et Iy.
#    Ses valeurs propres λ1 et λ2 révèlent la nature de chaque zone.
#    λ1 >> 0 ET λ2 >> 0 → coin détecté ✅
#
# 3. SCORE R = det(M) - k * trace(M)²
#    Évite de calculer les valeurs propres explicitement.
#    R >> 0 → coin / R << 0 → bord / |R| ≈ 0 → zone plate
#
# 4. PIPELINE :
#    a. Niveaux de gris
#    b. cv.cornerHarris → calcule R pour chaque pixel
#    c. Seuillage sur R → garde les vrais coins
#    d. Dilatation → agrandit les points pour la visualisation
#    e. Affinage sous-pixel
#    f. Affichage des coins sur l'image originale
# =============================================================

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'lighthouse.png')


# =========================
# 1. Chargement
# =========================

def load_image(path):
    """
    Charge l'image en BGR.
    Harris travaille sur les niveaux de gris mais on garde
    l'image couleur pour dessiner les coins en rouge dessus.
    """
    img = cv.imread(path)
    assert img is not None, f"Image non chargée : {path}"
    return img

def to_gray(img_bgr):
    """
    Convertit BGR → niveaux de gris.
    Harris a besoin des gradients d'intensité — un seul canal suffit.
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def to_rgb(img_bgr):
    """Convertit BGR → RGB pour affichage Matplotlib."""
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


# =========================
# 2. Calcul du score Harris
# =========================

def compute_harris(img_gray, block_size=2, ksize=3, k=0.04):
    """
    Calcule le score R de Harris pour chaque pixel de l'image.

    cv.cornerHarris fait tout en interne :
    1. Calcule les gradients Ix et Iy avec un filtre Sobel de taille ksize
    2. Calcule les produits Ix², Iy², IxIy
    3. Les lisse avec une fenêtre gaussienne de taille block_size
    4. Calcule R = det(M) - k * trace(M)² pour chaque pixel

    Le résultat est une image float32 de même taille que l'entrée
    où chaque valeur = le score R du pixel correspondant.

    R >> 0  → coin      → on va les garder
    R << 0  → bord      → on ignore
    |R| ≈ 0 → zone plate → on ignore

    Paramètres :
    block_size → taille de la fenêtre W pour calculer M (typiquement 2)
    ksize      → taille du filtre Sobel pour les gradients (3, 5 ou 7)
    k          → paramètre libre Harris (0.04 à 0.06)
                 plus grand = moins de coins détectés
    """
    img_float = np.float32(img_gray)
    harris_response = cv.cornerHarris(img_float, block_size, ksize, k)
    return harris_response


# =========================
# 3. Détection des coins
# =========================

def detect_corners(img_bgr, harris_response, threshold_ratio=0.01):
    """
    Garde uniquement les pixels dont le score R dépasse le seuil
    et les dessine en rouge sur une copie de l'image originale.

    threshold_ratio → fraction du score R maximum utilisée comme seuil
    threshold_ratio = 0.01 → garde les pixels dont R > 1% du max de R

    Plus threshold_ratio est petit → plus de coins détectés
    Plus threshold_ratio est grand → moins de coins, plus sélectif

    La dilatation agrandit chaque coin détecté pour le rendre
    visible à l'œil — sans ça les coins seraient des pixels isolés
    trop petits pour être vus.
    """
    img_corners = img_bgr.copy()

    harris_dilated = cv.dilate(harris_response, None)

    threshold = threshold_ratio * harris_dilated.max()
    img_corners[harris_dilated > threshold] = [0, 0, 255]

    n_corners = np.sum(harris_dilated > threshold)
    print(f"[Harris] Coins détectés : {n_corners} pixels")

    return img_corners, harris_dilated


# =========================
# 4. Affinage sous-pixel
# =========================

def refine_corners_subpixel(img_gray, harris_response,
                             threshold_ratio=0.01):
    """
    Affine la position des coins à la précision sous-pixel.

    cv.cornerHarris détecte les coins au niveau du pixel.
    cv.cornerSubPix les affine avec une précision plus fine
    en cherchant le minimum de la variation d'intensité dans
    un voisinage autour de chaque coin détecté.

    En vision par ordinateur (calibration caméra, reconstruction 3D,
    suivi d'objets), une précision au pixel entier ne suffit pas.
    cornerSubPix permet d'atteindre une précision de l'ordre de 0.1px.

    winSize   → demi-taille de la fenêtre de recherche (5,5) = fenêtre 11x11
    zeroZone  → zone morte au centre (-1,-1) = pas de zone morte
    criteria  → critère d'arrêt : max 30 itérations OU précision 0.001
    """
    threshold = threshold_ratio * harris_response.max()
    corners = np.argwhere(harris_response > threshold)

    corners_xy = np.float32([[c[1], c[0]] for c in corners])
    corners_xy = corners_xy.reshape(-1, 1, 2)

    if len(corners_xy) == 0:
        return corners_xy

    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30, 0.001
    )

    corners_refined = cv.cornerSubPix(
        img_gray, corners_xy,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    print(f"[SubPixel] {len(corners_refined)} coins affinés")
    return corners_refined


# =========================
# 5. Visualisation grille
# =========================

def show_images(images, titles, cmap_list=None):
    """Affiche une grille 2 x N."""
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
    # 1) Chargement
    img_bgr  = load_image(img_path)
    img_gray = to_gray(img_bgr)
    img_rgb  = to_rgb(img_bgr)

    # 2) Score Harris
    harris_response = compute_harris(img_gray, block_size=2, ksize=3, k=0.04)

    # 3) Détection et dessin des coins en rouge
    img_corners, harris_dilated = detect_corners(
        img_bgr, harris_response, threshold_ratio=0.01
    )
    img_corners_rgb = to_rgb(img_corners)

    # 4) Affinage sous-pixel — coins en vert
    corners_refined = refine_corners_subpixel(
        img_gray, harris_response, threshold_ratio=0.01
    )

    img_subpixel = img_bgr.copy()
    for corner in corners_refined:
        x, y = corner.ravel()
        cv.circle(img_subpixel, (int(x), int(y)), 3, (0, 255, 0), -1)
    img_subpixel_rgb = to_rgb(img_subpixel)

    # 5) Affichage — 3 images : original, coins, sous-pixel
    images = [
        img_rgb,
        img_corners_rgb,
        img_subpixel_rgb
    ]

    titles = [
        "Originale",
        "Coins détectés (rouge)",
        "Coins affinés sous-pixel (vert)"
    ]

    cmap_list = [None, None, None]

    show_images(images, titles, cmap_list=cmap_list)


if __name__ == "__main__":
    main()