# =============================================================
# DAY 11 — SEGMENTATION : Graph-Cut
# =============================================================
# CONCEPTS CLÉS À RETENIR :
#
# 1. QU'EST-CE QUE GRAPH-CUT ?
#    L'image = un graphe :
#    - Chaque pixel    = un nœud
#    - Source S        = objet
#    - Puits T         = fond
#    - Arêtes voisins  = terme frontière (similarité couleur)
#    - Arêtes pixel→ST = terme région   (coût objet/fond)
#
#    But : trouver la COUPE MINIMALE séparant S de T
#    → Min-Cut = Max-Flow (théorème fondamental)
#
# 2. ÉNERGIE À MINIMISER :
#    E(pixel) = R(pixel) + λ * B(frontière)
#    R = terme région    : -log P(couleur | GMM_objet ou GMM_fond)
#    B = terme frontière : exp(-||I(p)-I(q)||² / 2σ²)
#
# 3. PIPELINE :
#    a. Dessin des graines (souris)
#    b. Graph-Cut via cv.grabCut (Boykov-Kolmogorov)
#    c. Raffinement morphologique
#    d. Visualisation
#
# 4. DIFFÉRENCE AVEC LES AUTRES :
#    Otsu      → 1 seuil → 2 classes
#    Watershed → marqueurs → N objets
#    Graph-Cut → énergie globale minimisée → optimal mathématiquement
# =============================================================

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'football.jpg')


# =========================
# 1. Chargement
# =========================

def load_image(path):
    """
    Charge une image depuis le disque et la retourne en BGR.

    Graph-Cut a absolument besoin de la couleur pour fonctionner.
    Il construit deux modèles de couleur — un pour l'objet, un pour
    le fond. Sans couleur, ces deux modèles seraient identiques
    si les deux régions ont la même luminosité.
    """
    img = cv.imread(path)
    assert img is not None, f"Image non chargée : {path}"
    return img


def to_rgb(img_bgr):
    """
    Convertit une image BGR en RGB.

    OpenCV charge toujours en BGR mais Matplotlib affiche en RGB.
    Sans cette conversion, le rouge et le bleu sont inversés
    à l'affichage.
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)


# =========================
# 2. Dessin des graines
# =========================

def draw_seeds(img_bgr):
    """
    Ouvre une fenêtre interactive pour dessiner les graines.

    Graph-Cut a besoin de savoir avec certitude quels pixels
    appartiennent à l'objet et quels pixels appartiennent au fond.
    On lui donne cette information en dessinant des traits à la souris.

    Clic gauche maintenu = dessine en VERT  → pixels objet certain
    Clic droit  maintenu = dessine en ROUGE → pixels fond certain

    Règle importante :
    Dessine sur le CŒUR des régions, jamais sur les bords.
    Les bords sont ambigus — mi-objet mi-fond — et pollueraient
    les modèles de couleur de Graph-Cut.

    Touches disponibles :
    'n' → valide les graines et lance Graph-Cut
    'r' → efface tout et recommence depuis zéro
    'q' → ferme la fenêtre sans lancer
    """
    h, w = img_bgr.shape[:2]

    # mask est une image de même taille que l'original
    # chaque pixel vaut 0 par défaut (= non dessiné)
    # quand on dessine, on écrit GC_FGD (objet) ou GC_BGD (fond)
    mask = np.zeros((h, w), np.uint8)

    # img_display est la copie sur laquelle on dessine les traits
    # on ne touche jamais à img_bgr original
    img_display = img_bgr.copy()

    # drawing = True quand le bouton souris est maintenu enfoncé
    # mode = 'fg' (objet) ou 'bg' (fond) selon le bouton cliqué
    drawing = False
    mode = 'fg'

    def draw(event, x, y, flags, param):
        """
        Fonction appelée automatiquement à chaque événement souris.

        OpenCV appelle cette fonction tout seul quand :
        - tu cliques (EVENT_LBUTTONDOWN ou EVENT_RBUTTONDOWN)
        - tu bouges la souris (EVENT_MOUSEMOVE)
        - tu relâches le bouton (EVENT_LBUTTONUP ou EVENT_RBUTTONUP)

        x, y = coordonnées du curseur dans l'image au moment de l'événement.

        nonlocal drawing, mode → on modifie les variables drawing et mode
        définies dans draw_seeds(), pas des nouvelles variables locales.
        """
        nonlocal drawing, mode

        # Clic gauche enfoncé → on passe en mode dessin objet
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            mode = 'fg'

        # Clic droit enfoncé → on passe en mode dessin fond
        elif event == cv.EVENT_RBUTTONDOWN:
            drawing = True
            mode = 'bg'

        # Souris qui bouge avec un bouton enfoncé → on dessine
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            if mode == 'fg':
                # Cercle vert sur l'image d'affichage — juste pour visualiser
                cv.circle(img_display, (x, y), 8, (0, 255, 0), -1)
                # Même cercle sur le masque mais avec la valeur GC_FGD (1)
                # C'est ce masque que Graph-Cut lira — pas l'image d'affichage
                cv.circle(mask, (x, y), 8, cv.GC_FGD, -1)
            else:
                # Même logique pour le fond — rouge sur l'affichage, GC_BGD sur le masque
                cv.circle(img_display, (x, y), 8, (0, 0, 255), -1)
                cv.circle(mask, (x, y), 8, cv.GC_BGD, -1)

        # Bouton relâché → on arrête de dessiner
        elif event in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
            drawing = False

    # Crée la fenêtre OpenCV avec le titre donné
    # Sans namedWindow, setMouseCallback ne sait pas sur quelle fenêtre s'accrocher
    window_name = 'Dessine — gauche:objet  droit:fond  n:lancer  r:reset  q:quitter'
    cv.namedWindow(window_name)

    # Branche la fonction draw() sur la fenêtre
    # À partir de maintenant, chaque clic ou mouvement souris dans cette fenêtre
    # appellera automatiquement draw() avec les coordonnées du curseur
    cv.setMouseCallback(window_name, draw)

    print("\n[Graph-Cut] Instructions :")
    print("  Clic gauche = OBJET (vert)  — cœur du ballon")
    print("  Clic droit  = FOND  (rouge) — tissu bleu loin du ballon")
    print("  'n' = lancer  |  'r' = reset  |  'q' = quitter\n")

    # Boucle infinie qui maintient la fenêtre ouverte
    # Sans cette boucle, la fenêtre s'ouvrirait et se fermerait instantanément
    # cv.waitKey(1) attend 1ms — assez court pour que l'interface reste fluide
    # & 0xFF → garde uniquement les 8 derniers bits de la touche pressée
    #           nécessaire sur certains systèmes pour avoir le bon code ASCII
    while True:
        # Affiche l'image d'affichage mise à jour dans la fenêtre
        cv.imshow(window_name, img_display)
        k = cv.waitKey(1) & 0xFF

        if k == ord('r'):
            # Remet tous les pixels du masque à 0
            mask[:] = 0
            # Recopie l'image originale pour effacer les traits dessinés
            img_display[:] = img_bgr.copy()
            print("[Reset] ✅")

        elif k == ord('n'):
            # Vérifie qu'on a au moins une graine objet ET une graine fond
            # Si l'une des deux manque, Graph-Cut ne peut pas construire ses GMM
            if not np.any(mask == cv.GC_FGD) or not np.any(mask == cv.GC_BGD):
                print("⚠️  Dessine des graines OBJET (gauche) ET FOND (droit) !")
                continue
            print("[Graines validées] ✅")
            break

        elif k == ord('q'):
            # Ferme toutes les fenêtres OpenCV et retourne None
            cv.destroyAllWindows()
            return None, None

    # Ferme la fenêtre de dessin avant de passer à la suite
    cv.destroyAllWindows()
    return mask, img_display


# =========================
# 3. Graph-Cut
# =========================

def apply_graphcut(img_bgr, seed_mask, n_iter=10):
    """
    Applique l'algorithme Graph-Cut via cv.grabCut en mode masque.

    On utilise GC_INIT_WITH_MASK — c'est le mode qui utilise
    nos graines dessinées à la souris. C'est plus précis que
    le mode rectangle du Day 09 car on contrôle exactement
    quels pixels sont objet et quels pixels sont fond.

    En interne, cv.grabCut implémente l'algorithme de Boykov-Kolmogorov :
    1. Construit les GMM fond/objet depuis les graines
    2. Construit le graphe S-T avec les arêtes entre pixels
    3. Calcule le Max-Flow pour trouver le Min-Cut optimal
    4. Met à jour les GMM avec les nouveaux labels trouvés
    5. Répète tout ça n_iter fois

    Les pixels non dessinés sont initialisés à GC_PR_BGD (fond probable)
    et non GC_BGD (fond certain) — comme ça Graph-Cut peut les reclasser
    librement selon ce que ses GMM décident.

    Après cv.grabCut, chaque pixel du masque a l'une de ces 4 valeurs :
    GC_BGD (0)    = fond certain    → on ne reviendra pas dessus
    GC_FGD (1)    = objet certain   → nos graines vertes
    GC_PR_BGD (2) = fond probable   → classé fond par Min-Cut
    GC_PR_FGD (3) = objet probable  → classé objet par Min-Cut
    """
    # np.where remplace chaque 0 du masque par GC_PR_BGD (2)
    # Les pixels dessinés (GC_FGD=1 ou GC_BGD=0 non nul) sont conservés
    working_mask = np.where(
        seed_mask == 0, cv.GC_PR_BGD, seed_mask
    ).astype(np.uint8)

    # bgd_model et fgd_model = tableaux internes qu'OpenCV utilise
    # pour stocker les 5 gaussiennes de chaque GMM (fond et objet)
    # On les initialise à zéro — OpenCV les remplit lui-même
    # Taille fixe (1, 65) en float64 — ne jamais modifier
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Lance Graph-Cut — modifie working_mask en place
    # None à la place du rect car on est en mode masque (pas rectangle)
    cv.grabCut(
        img_bgr,
        working_mask,
        None,
        bgd_model,
        fgd_model,
        n_iter,
        cv.GC_INIT_WITH_MASK
    )

    # Construit le masque binaire final :
    # objet certain (1) ET objet probable (3) → 255 (blanc)
    # fond certain  (0) ET fond probable  (2) → 0   (noir)
    mask_binary = np.where(
        (working_mask == cv.GC_FGD) | (working_mask == cv.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    return mask_binary, working_mask


# =========================
# 4. Raffinement
# =========================

def refine_mask(mask_binary):
    """
    Nettoie le masque brut que Graph-Cut a produit.

    Graph-Cut peut laisser trois types de défauts :
    - Des trous à l'intérieur de l'objet segmenté
    - De petits îlots de bruit isolés dans le fond
    - Des bords en escalier irréguliers

    On corrige ça en trois étapes :
    1. Fermeture morphologique (dilatation puis érosion)
       → comble les petits trous à l'intérieur de l'objet
    2. Ouverture morphologique (érosion puis dilatation)
       → supprime les petits composants isolés dans le fond
    3. Flou gaussien + seuillage
       → arrondit et lisse les bords irréguliers
    """
    # Fermeture — kernel elliptique 15x15, assez grand pour combler les trous
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    mask = cv.morphologyEx(mask_binary, cv.MORPH_CLOSE, kernel_close)

    # Ouverture — kernel plus petit pour ne pas déformer l'objet
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

    # Flou gaussien sur le masque converti en float32
    # → produit des valeurs entre 0 et 255 avec des transitions douces
    # Seuil à 100 → garde uniquement les zones bien blanches
    # * 255 → remet en uint8 avec des valeurs 0 ou 255
    blurred = cv.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    mask = (blurred > 100).astype(np.uint8) * 255

    return mask


# =========================
# 5. Visualisation labels
# =========================

def visualize_labels(mask_after):
    """
    Crée une image colorée qui montre la décision de Graph-Cut
    pour chaque pixel de l'image.

    Rouge = GC_BGD (0)    fond certain    → nos graines rouges
    Vert  = GC_FGD (1)    objet certain   → nos graines vertes
    Bleu  = GC_PR_BGD (2) fond probable   → classé fond par Min-Cut
    Cyan  = GC_PR_FGD (3) objet probable  → classé objet par Min-Cut

    La frontière entre le bleu et le cyan est exactement
    la coupe minimale trouvée par l'algorithme — c'est la ligne
    qui sépare l'objet du fond avec le coût d'énergie le plus faible.
    """
    h, w = mask_after.shape

    # Crée une image noire vide de même taille, 3 canaux couleur
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Colorie chaque zone selon son label
    viz[mask_after == cv.GC_BGD]    = [255, 0,   0  ]   # rouge
    viz[mask_after == cv.GC_FGD]    = [0,   255, 0  ]   # vert
    viz[mask_after == cv.GC_PR_BGD] = [0,   0,   255]   # bleu
    viz[mask_after == cv.GC_PR_FGD] = [0,   255, 255]   # cyan

    return viz


# =========================
# 6. Visualisation grille
# =========================

def show_images(images, titles, cmap_list=None):
    """
    Affiche toutes les images dans une grille 2 x N.

    (n+1)//2 calcule le nombre de colonnes nécessaires
    pour ranger n images sur 2 lignes.
    zip() associe chaque image à son titre et sa cmap.
    cmap='gray' force l'affichage en niveaux de gris.
    cmap=None laisse Matplotlib choisir (RGB si 3 canaux).
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
# 7. Pipeline complet
# =========================

def main():
    # Charge l'image depuis le disque
    img_bgr = load_image(img_path)
    img_rgb  = to_rgb(img_bgr)

    # Ouvre la fenêtre de dessin et attend les graines
    seed_mask, img_seeds_display = draw_seeds(img_bgr)
    if seed_mask is None:
        return
    img_seeds_rgb = to_rgb(img_seeds_display)

    # Lance Graph-Cut avec les graines dessinées
    mask_binary, mask_after = apply_graphcut(img_bgr, seed_mask, n_iter=10)

    # Nettoie le masque brut
    mask_refined = refine_mask(mask_binary)

    # Applique le masque sur l'image originale pour extraire l'objet
    # bitwise_and : garde les pixels de l'objet, met le fond à noir
    img_result     = cv.bitwise_and(img_bgr, img_bgr, mask=mask_refined)
    img_result_rgb = to_rgb(img_result)

    # Construit la carte des labels pour visualisation
    label_viz = visualize_labels(mask_after)

    images = [
        img_rgb,
        img_seeds_rgb,
        label_viz,
        mask_binary,
        mask_refined,
        img_result_rgb
    ]

    titles = [
        "Originale",
        "Graines (vert=objet, rouge=fond)",
        "Labels Graph-Cut",
        "Masque brut",
        "Masque raffiné",
        "Objet extrait"
    ]

    cmap_list = [None, None, None, 'gray', 'gray', None]

    show_images(images, titles, cmap_list=cmap_list)


if __name__ == "__main__":
    main()