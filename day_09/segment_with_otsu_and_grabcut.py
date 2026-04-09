# Day 9 : SEGMENTATION : Otsu and GrabCut
# Suite à une contrainte de déplacement (voyage), ce sera enregistrée sans voix. Merci de comprendre
"""
Concepts clés à retenir :
1. Otsu => segmentation AUTOMATIQUE par seuillage
- Travaille sur l'histogramme(niveau de gris)
- Trouve le seuil qui maximise la variance inter-classes
- Hypothèse forte : histogramme BIMODALE (2 pics distincts)
- Resultat : image BINAIRE (noir/blanc)
2. GRABCUT => SEGMENTATION SEMI-AUTOMATIQUE par graphe
- Travaille sur la couleur (BGR) - pas les niveaux de gris
- L'utilisateur donne un rectangle englobant l'objet
- GrabCut modélise fond/objet avec des GMM (mélanges gaussiens)
- Minimise une énergie globale : région + frontière
- Résultat : masque binaire objet/fond

DIFFERENCE FONDAMENTALE :
Otsu => 100% automatique, mais limité aux images simples
Grabut => semi- automatique, mais bien plus lent précis et flexible
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# chemin dossier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'football.jpg')

# 1. chargement de l'image
def load_image(path):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError("impossible de charger le fichier")
    return img

def to_gray(img_bgr):
    """
    convertir BGR => niveau de gris
    Otsi travaille sur un canal d'intensité. ON ne peut pas appliquer Otsu directement sur une image couleur
    """
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def to_rgb(img_bgr):
    """convertir BGR en RGB pour affichage Matplotlib"""
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# 2. Segmentation par Otsu
def segment_otsu(img_gray):
    """seuillage d'Otsu => trouve automatiquement le seuil optimal.
    Otsu parcourt tous les seuils possibles(0 à 255) et cherche celui qui maximise la variance inter-classes

    Paramètres   : img_gray : image en niveaux de gris (obligatoire)
                 0        : seuil ignoré, Otsu le calcule lui-même
                 255      : valeur max assignée aux pixels > seuil
                 THRESH_BINARY : pixels > seuil -> 255, sinon -> 0
                 THRESH_OTSU : active le calcul automatique du sueil
    LIMITATIONS IMPORTANTE :
    Otsu suppose un histogramme BIMODAL (deux pics bien séparés). Si l'image a un éclairage non uniforme ou plusieurs
    objets, le seuil trouvé sera sous-optimal -> c'est pourquoi il faut utiliser GrabCut à la place
                 """
    th,img_otsu = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    print(f"[Otsu] seuil optimal : {th:.1f}")
    return img_otsu,th

def segment_otsu_with_blur(img_gray,ksize=(5,5)):
    """Otsu + flou gaussien préalable
    Règle importante : image bruitée -> toujours appliquer flou avant Otsu
                       image propre -> Otsu seul suffit"""
    img_blur = cv.GaussianBlur(img_gray,ksize,0)
    th,img_otsu = cv.threshold(img_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    print(f"[Otsu + flou] seuil optimal : {th:.1f}")
    return img_otsu,th

# Selection intérative interactive du rectange (GrabCut)
def select_rect_interactif(img_bgr):
    """
    ouvre une fenetre Opencv pour dessiner le rectangle à la souris.
    Comment utiliser selectROI :
    1. une fenetre s'ouvre avec ton image
    2. clique et glisse pour dessiner le rectangle
    3. appuie sur entrer ou espace pour valider
    4. appuie sur C pour annuler ou recommencer

    paramètres selectROI :
    - fromCenter = false : le rectangle part du coin supérieur gauche (plus intuitif que de partir du centre)
    - showCrosshair = True : affiche un réticule pour aider le placement

    # Regle fondamentale :
    ->tout ce qui est en dehors du rectangle = fond certain
        GrabCut ne reviendra jamais sur ces pixels
    -> Trop serre = perd des parties ce l'objet
    -> Trop large = trop de fond, GMM mal initailisé

    Retourne : (x,y,w,h) - coordonnées du rectangle selectionné
    """
    rect = cv.selectROI("Dessine le rectange puis appuie sur ENTRER",img_bgr,fromCenter=False,showCrosshair=True)
    cv.destroyAllWindows()
    print(f"[selectROI] Rectangle selectionné : x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}:")
    return rect

# Segmentation par GrabCut

def segment_grabcut(img_bgr,rect,n_iter=5):
    """
    GrabCut segmentation semi-automatique par graph cut
    :param img_bgr: image couleur BGR - GrabCut exploite la couleur (contrairement à Otsu qui travaille en gray
    :param rect: (x,y,w,h) - rectangle dessiné par l'utilisateur
    :param n_iter: nombre d'itération _5 est suffisant est général. Plus d'itération = meilleur resultat mais plus lent
    a retenir : masque GrabCut (4 valeurs possibles):
    cv.GC_BGD (0) = fond certain -> pixels hors rectangle
    cv. GC_FGD (1) = objet certain -> marqué manuellement
    cv.GC_PR_BGD (2) = fond probable -> décidé par l'algo
    cv.GC_PR_FGD (3) = objet probable -> decidé par l'algo

    # a retenir : bgd_model et fgd_model
    Tableaux internes requis par opencv pour stocker les GMM. Toujours initialisés à zero, taille fixe (1,65) en float64.
    Ne pas y toucher -openCV les gère lui même


    # Limitation :
    GrabCut peut rater si fond et objet ont des couleurs similaires
    Dans ce cas, augmenter n_iter ou affiner le rectangle
    """

    # Masque initialisé à zero (tout = fond probable)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    # modèles GMM internes - requis par OpenCV, ne pas modifier
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    # lancement de GrabCut
    # GC_INIT_WITH_RECT = initialisation par rectangle
    cv.grabCut(img_bgr, mask, rect, bgd_model, fgd_model,n_iter,cv.GC_INIT_WITH_RECT)
    # construire le masque binaire
    # objet certain (1) ou objet probable (3) -> 255
    # fond certain (.) ou fond probable(2) -> .
    mask_binary = np.where((mask==cv.GC_FGD) | (mask==cv.GC_PR_FGD),255,0).astype(np.uint8)
    # appliquer le masque sur l'image originale
    # bitwise_and : garde les pixels de l'objet , met le fond à 0
    img_result = cv.bitwise_and(img_bgr,img_bgr,mask=mask_binary)
    return mask_binary,img_result

# Visulation

def show_images(images,titles,cmap_list=None):
    """
    affiche une grille 2 x N
    :cmap_list: permet de mixer images couleurs et niveaux de gris dans la même grille sans deux fonctions séparées
    :param titles:
    :None: Matplotlib choisit automatiquement (RGB si 3 canaux)
    :gray: force l'affichage en niveaux de gris (1 canal)
    """
    n = len(images)
    cols = (n+1) // 2
    rows = 2

    if cmap_list is None:
        cmap_list = ['gray']*n
    plt.figure(figsize=(4*cols,8))

    for i, (img,title,cmap) in enumerate(zip(images,titles,cmap_list)):
        plt.subplot(rows,cols,i+1)
        plt.imshow(img,cmap=cmap)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# pipeline complet

def main():
    # 1 chargment
    img_bgr = load_image(img_path)
    img_gray = to_gray(img_bgr)
    img_rgb = to_rgb(img_bgr)

    # 2 Otsu simple
    img_otsu,th = segment_otsu(img_gray) # prend une image en niveau de gris
    # 3 Otsu + flou gaussien
    img_otsu_blur,th2 = segment_otsu_with_blur(img_gray,ksize = (5,5))

    # 4 selection interactive du rectangle pour GrabCut
    rect = select_rect_interactif(img_bgr)
    # 5 GrabCut avec le rectangle selectionné
    mask_gc, img_gc = segment_grabcut(img_bgr,rect=rect,n_iter=5) #image lue en bgr
    img_gc_rgb = to_rgb(img_gc) # après on la convertir en rgb

    # 6 affichage complet
    images = [img_rgb,img_otsu,img_otsu_blur,mask_gc,img_gc_rgb]
    titles = ["originale",f"Otsu (seuil={th:.0f}",f"Otsu + flou (seuil={th2:.0f}","masque GrabCut","GrapCut -objet extrait"]
    cmap_list = [None,'gray','gray','gray',None]
    show_images(images,titles,cmap_list=cmap_list)

if __name__ == "__main__":
    main()