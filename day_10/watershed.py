# Day 10 : Segmentation Watershed Algorithm

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# chemin dossier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'coins1.png')

# 1. chargement

def load_image(path):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"impossible de charger : {path}")
    return img

def to_gray(img_bgr):
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def to_rgb(img_bgr):
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# 2. Binarization (otsu inversé)
def binarize(img_gray):
    #on va utiliser "Binarization par Otsu inversé "
    """on inverse pour avoir : objets = blanc (255) et fond = noir (0)
    c'est la convention attendue pour les opérations morphologiques (erosion, dilation)
    la distance transform (travaille sur les pixels blancs)"""

    th, img_bin = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img_bin

# 3. Nettoyage morphologique
# ouverture morphologique pour supprimer le bruit
def clean_binary(img_bin,ksize=3,iterations = 2):
    """ouverture = erosion + dilatation
       erosion = supprime les petits objets objets parasites
       dilatation = restaure la taille des vrais objets

       pourquoi faut-il le faire ici ?
       parce que, le bruit crée de faux marqueurs => de fausses frontières (segmentation incorrecte)
       ksize = (3 = leger et 5 un peu aggressif)
       iteration = nbre de fois il faut faire l'ouverture
       """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    img_clean = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel,iterations=iterations)
    return img_clean

# 4. Fond correct
def get_sure_background(img_clean,ksize=3,iterations=2):
    """en dilatant les objets, on pousse leurs bords vers l'exterieur. Ce qui reste en dehors de cette
    dilatation = fond certain à 100%"""

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))
    sure_bg = cv.dilate(img_clean, kernel, iterations=iterations)
    return sure_bg

# 5. Objet certain
def get_sure_foreground(img_clean,threshold_ratio=0.5):
    """Distance Transform pour trouver l'objet certain
    La distance Transforme c'est quoi ? pour chaque pixel blanc (objet),calcule sa distance au pixel Noir (fond) le
    plus proche.
    centre de l'objet = distance maximale (loin des bords) et bord de l'objet = distance faible (proche du fond)

    threshold ratio = garde uniquement les pixels dont dist > ratio * dist_max
    0.5 garde les pixels à plus de de 50%  de la distance max.
    Ration eleve => zone certaine plus petite mais plus faible
    Ration faible => zone certaine plus grande mais moins faible"""
    """Paramètres de distance transform : 
       DIST_L2 : distance euclidienne _- la plus precise
       maskSize = 5 : taille du masque de calcul (3 ou 5)"""
    dist_transform = cv.distanceTransform(img_clean, cv.DIST_L2, maskSize=5)
    # seuillage : garde uniquement les centres des objets
    val,sure_fg = cv.threshold(dist_transform,threshold_ratio* dist_transform.max(),255,cv.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    return sure_fg,dist_transform

# 6. zone inconnue
def get_unknown_zone(sure_bg,sure_fb):
    """zone inconnue = fond certain - objet certain
    """
    inconnu = cv.subtract(sure_bg,sure_fb)
    return inconnu

# 7. marqueurs pour watershed

def create_markers(sure_fg,unknown):
    """créer les marques labelisés pour watershed
    Pourquoi ? Watershed a besoin de "graines" pour savoir où commencer à remplir chaque bassin.
    ces graines = marqueurs labelisés

    la methode connectedComponents : attribue un label unique à chaque comosante connexe de sure_fg
    label 0 = fond
    label 1,2,... = objets individuels détéctés
    Pourquoi +1 sur tous les LABELS ?
    watershed reserve 0 pour la zone inconnue, sans le +1 , le fond (label 0) serait confondu avec la zone inconnue"""
    """Après +1 :
    fond = label 1
    objet 1 = label 2
    objet 2 = label 3...
    
    on remet à 0 les pixels de la zone inconnue
    watershed sait qu'il doit decider pour ces pixels"""
    val,markers = cv.connectedComponents(sure_fg)
    # Decaler - libère le label 0 pour la zone inconnue
    markers = markers + 1

    # zone inconnue - 0 ( watershed va decaler)
    markers[unknown==255] = 0

    return markers

# 8. appliquer watershed
def apply_watershed(img_bgr,markers):
    """A retenir : ce que fait Watershed :
    1. Part des marquers (graines) de chaque bassin
    2. remplit progressivement chaque bassin vers l'exterieur
    3. là où deux bassins se rejoignent = frontière
    4. marque ces frontières avec la valeur -1 dans markers

    Paramètres de la methode watershed :
    img_bgr : image couleur BGR obligatoire
    markers : carte des markeurs int32 _ obligatoire

    Après WATERSHED :
    markers == -1 -> pixels de frontière entre deux objets
    On les colore en rouge (0,0,255) en BGR pour les visauliser

    LIMITATIONS :
    Watershed est sensible au bruit -> toujours prétraiter
    Sur-segmentation possible si les marqueurs sont trop nombreux"""
    img_ws = img_bgr.copy()
    markers = markers.astype(np.int32)

    cv.watershed(img_bgr,markers)

    # Frontières = -1 -> rouge en BGR
    img_ws[markers == -1] = [0,0,255]
    return img_ws,markers

# 9. visualisation
def show_images(images,titles,cmap_list=None):
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
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 10. Pipeline complet
def main():
    # 1. chargement
    img_bgr = load_image(img_path)
    img_gray = to_gray(img_bgr)
    img_rgb = to_rgb(img_gray)
    # 2. binarisation
    img_bin = binarize(img_gray)

    # 3. Nettoyage morphologique - supprime le bruit
    img_clean = clean_binary(img_bin,ksize=3,iterations=2)

    # 4. Fond certain - Dilatation
    sure_bg = get_sure_background(img_clean,ksize=3,iterations=5)

    # 5. Objet certain - Distance Transfor
    sure_fg, dist_transform = get_sure_foreground(img_clean,threshold_ratio=0.7)

    # 6. Zone inconnue - Soustraction
    unknown = get_unknown_zone(sure_bg,sure_fg)

    # 7. Markeurs labelisés
    markers = create_markers(sure_fg,unknown)

    # 8. Watershed
    img_watershed, img_result = apply_watershed(img_bgr,markers)
    img_watershed_rgb = to_rgb(img_watershed)

    # Distance transform normalisée pour affichage
    dist_display = (dist_transform / dist_transform.max()*255).astype(np.uint8)

    images = [img_rgb,img_bin,img_clean,sure_fg,dist_display,unknown,img_watershed_rgb]
    titles = ["Originale","binarisation Otsu(INV)","Nettoyage morpho","Objet certain","Distance Transform","Zone inconnue","Watershed-frontières"]

    cmap_list = [None,'gray','gray','gray','gray','gray',None]
    show_images(images,titles,cmap_list=cmap_list)

if __name__ == '__main__':
    main()