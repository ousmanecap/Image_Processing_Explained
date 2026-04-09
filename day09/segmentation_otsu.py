# SEGMENTATION
# Otsu Thresholding (global automatique)
# GrabCut (segmentation interactive par graphe)

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Chemin et chargement
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'football.jpg')

# =========================
# 1. Chargement & Prétraitement
# =========================

def load_image(path):
    """
    Charger une image en BGR.
    """
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {path}")
    return img

def to_gray(img_bgr):
    """
    Convertir l'image BGR en niveau de gris.
    Nécessaire pour Otsu — travaille sur un seul canal.
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def to_rgb(img_bgr):
    """
    Convertir BGR en RGB pour affichage Matplotlib.
    OpenCV charge en BGR, Matplotlib attend du RGB.
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# =========================
# 2. Segmentation par Otsu
# =========================

def segment_otsu(img_gray):
    """
    Applique le seuillage d'Otsu pour segmenter automatiquement.

    Otsu cherche le seuil t* qui maximise la variance inter-classes :
        sigma_B²(t) = w0(t) * w1(t) * [mu0(t) - mu1(t)]²

    où :
        w0, w1 = probabilités des deux classes (fond / objet)
        mu0, mu1 = moyennes d'intensité de chaque classe

    OpenCV retourne (seuil_trouvé, image_binarisée).
    Le 0 passé comme seuil est ignoré — Otsu le calcule automatiquement.
    """
    otsu_thresh, img_otsu = cv.threshold(
        img_gray,
        0,           # valeur ignorée, Otsu calcule le seuil optimal
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    print(f"[Otsu] Seuil optimal trouvé : {otsu_thresh:.1f}")
    return img_otsu, otsu_thresh

def segment_otsu_with_blur(img_gray, ksize=(5, 5)):
    """
    Otsu précédé d'un flou gaussien pour réduire le bruit.
    Le flou lisse l'histogramme et rend les deux pics plus nets —
    Otsu trouve un seuil plus robuste.
    """
    img_blur = cv.GaussianBlur(img_gray, ksize, 0)
    otsu_thresh, img_otsu_blur = cv.threshold(
        img_blur,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    print(f"[Otsu + flou] Seuil optimal trouvé : {otsu_thresh:.1f}")
    return img_otsu_blur, otsu_thresh

# =========================
# 3. Segmentation par GrabCut
# =========================

def segment_grabcut(img_bgr, rect=None, n_iter=5):
    """
    Applique GrabCut pour segmenter un objet du fond.

    GrabCut est un algorithme de segmentation basé sur les graphes (graph cut).
    Il modélise le fond et l'objet comme des mélanges gaussiens (GMM)
    et minimise une énergie globale :

        E = lambda * R(alpha) + B(alpha)

    où :
        R(alpha) = terme région (probabilité GMM de chaque pixel)
        B(alpha) = terme frontière (pénalité de discontinuité entre voisins)
        lambda   = poids relatif des deux termes

    Fonctionnement :
    1. L'utilisateur fournit un rectangle englobant l'objet (rect)
    2. GrabCut initialise les GMM fond/objet
    3. Il alterne entre :
       - mise à jour des GMM (EM)
       - résolution du min-cut pour trouver la segmentation optimale
    4. On itère n_iter fois

    Paramètres :
        img_bgr : image couleur BGR (GrabCut utilise la couleur)
        rect : (x, y, w, h) rectangle englobant l'objet — fourni manuellement
        n_iter : nombre d'itérations (5 est généralement suffisant)

    Retourne :
        mask_binary : masque binaire 0/255 (objet = 255, fond = 0)
        img_result  : image originale avec fond supprimé
    """
    if rect is None:
        # Rectangle par défaut : 10% de marge sur chaque bord
        h, w = img_bgr.shape[:2]
        margin_x = w // 10
        margin_y = h // 10
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    # Masque GrabCut — initialisé à 0 (tout = fond probable)
    # Valeurs possibles :
    #   cv.GC_BGD (0)    = fond certain
    #   cv.GC_FGD (1)    = objet certain
    #   cv.GC_PR_BGD (2) = fond probable
    #   cv.GC_PR_FGD (3) = objet probable
    mask = np.zeros(img_bgr.shape[:2], np.uint8)

    # Modèles GMM internes — tableaux temporaires requis par OpenCV
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, n_iter, cv.GC_INIT_WITH_RECT)

    # Pixels certains ou probables objet → 1, reste → 0
    mask_binary = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Appliquer le masque sur l'image originale
    img_result = cv.bitwise_and(img_bgr, img_bgr, mask=mask_binary)

    return mask_binary, img_result

# =========================
# 4. Visualisation
# =========================

def show_images(images, titles, cmap_list=None):
    """
    Affiche une grille 2 x N.
    cmap_list : liste de cmaps par image (None = 'gray' par défaut)
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
# 5. Pipeline complet
# =========================

def main():
    
    img_bgr = load_image(img_path)
    img_gray = to_gray(img_bgr)
    img_rgb = to_rgb(img_bgr)

    # 2) Otsu simple
    img_otsu, thresh1 = segment_otsu(img_gray)

    # 3) Otsu + flou gaussien
    img_otsu_blur, thresh2 = segment_otsu_with_blur(img_gray, ksize=(5, 5))

    # 4) GrabCut
    mask_gc, img_gc = segment_grabcut(img_bgr, rect=None, n_iter=5)
    img_gc_rgb = to_rgb(img_gc)  # pour affichage Matplotlib

    # 5) Affichage comparatif
    images = [
        img_rgb,
        img_otsu,
        img_otsu_blur,
        mask_gc,
        img_gc_rgb
    ]

    titles = [
        "Originale",
        f"Otsu (seuil={thresh1:.0f})",
        f"Otsu + flou (seuil={thresh2:.0f})",
        "Masque GrabCut",
        "GrabCut — objet extrait"
    ]

    cmap_list = [None, 'gray', 'gray', 'gray', None]

    show_images(images, titles, cmap_list=cmap_list)

if __name__ == "__main__":
    main()