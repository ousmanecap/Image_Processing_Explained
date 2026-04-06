
# Hough Transform
# -- Hough lines et Circles

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

#les chemins et chargemnt de l'image
BASE_DIR = os.getcwd()
img_path = os.path.join(BASE_DIR, 'imdata', 'circles.png')

img_color = cv.imread(img_path) #lit notre image BGR
assert img_color is not None, f"image non chargeée : {img_path}"

img_gray = cv.cvtColor(img_color,cv.COLOR_BGR2GRAY)

#Pretraitement - Canny

def get_edges(img_gray,low=50,high=150):
    """
    appliquer un lissage aussien puis canny pour extraire les contours.  
    Hough travaille sur des pixels de contours, pas sur l'image brute
    """

    blur = cv.GaussianBlur(img_gray,(5,5),sigmaX=0)
    edges = cv.Canny(blur,low,high)
    return edges 

#Hough lines - droites infinies (methode classique)

def hough_lines_standard(img,edges,threshold=150):
    """ detecte des droites infinies via la transformée de Hough classique
        chaque pixel de contour (x,y) vote pour toutes les droites passant par lui dans l'espace polaire :
    """
    img_copy = img.copy() # pour copier notre image originale afin de dessiner les lignes dessus
    lines = cv.HoughLines(edges,rho=1, theta=np.pi/180,threshold=threshold)

    if lines is not None:
        for line in lines:
            rho,theta = line[0]
            a,b = np.cos(theta),np.sin(theta)  
            x0,y0 = a * rho, b* rho

            # Pointes à +- 1000px pour simuler une droite infiie
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000* (a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000* (a))

            cv.line(img_copy,(x1,y1),(x2,y2),(0,0,255),2)

    return img_copy

# Hough linesP - segments finis (methode probabilisite)

def hough_lines_proba(img,edges,threshold=100,min_line_length = 50,max_line_gap=10):
    """
    detecter des segments fins via la transformée de hough probabiliste
    # différence clé avec la méthode standard:
    - tire aléatoirement des sous-ensembles de pixels -> plus rapide
    - retourne directement (x1,y1,x2,y2) -> pas de reconversion necessair
    Paramètres :
    threshold :vote minimum pour valider un segment
    min_line_length = longueur minimale accepte (pixels)
    max_line_gap : espace max pour fusionner deux segments 
    """
    img_copy = img.copy()
    linesP = cv.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=threshold,minLineLength=min_line_length,maxLineGap=max_line_gap)

    if linesP is not None:
        for line in linesP:
            x1,y1,x2,y2 = line[0]
            cv.line(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
    return img_copy

# Hough Circles

def hough_circles(img, img_gray,dp=1,min_dist=30,param1=50,param2=30,min_radius=0,max_radius=0):
    # detecter les cercles

    # Attention prend le grayscale en entrée (pas les contours de Canny)
    # il applique Canny en interne
    """ paramètres :
    dp = ration resolution accumulateur / reoslution image
    min_dist = distance minimale entre deux cercles detectés
    param1 = seuil haut de canny inerne
    param2 = seuil accumulateur - plus bas = 'plus de detection'
    min_radius = rayon minimum accepté (0 = pas de limite)
    max_radius = rayon maximum acccepté (0 = pas de limite)
    """
    img_copy = img.copy()

    # leger flou avant Hough circles pour reduire les faux contours
    blur = cv.GaussianBlur(img_gray,(9,9),sigmaX=2)
    circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,dp=dp,minDist=min_dist,param1=param1,param2=param2,minRadius=min_radius,maxRadius=max_radius)

    if circles is not None:
        circles_int = np.round(circles[0]).astype(int)
        for x, y,r in circles_int:
            cv.circle(img_copy,(x,y),r,(255,0,0),2)  # cercle
            cv.circle(img_copy,(x,y),2,(0,0,255),3) # centre
    return img_copy

# main

if __name__=="__main__":
    edges = get_edges(img_gray,low=50,high=150)

    img_std = hough_lines_standard(img_color,edges,threshold=150)

    img_prob = hough_lines_proba(img_color,edges,threshold=100,min_line_length=50,max_line_gap=10)

    img_cir = hough_circles(img_color,img_gray,dp=1,min_dist=30,param1=50,param2=20,min_radius=0,max_radius=0)


    # liste pour les images et les titres

    images = [cv.cvtColor(img_color, cv.COLOR_BGR2RGB),edges,cv.cvtColor(img_std, cv.COLOR_BGR2RGB),cv.cvtColor(img_prob, cv.COLOR_BGR2RGB),cv.cvtColor(img_cir, cv.COLOR_BGR2RGB)]

    # titres
    titles = ["image origninale","contours Canny","HoughLines (standard)","HoughLinesP(Proba)","HoughCircles"]

    fig, axes = plt.subplots(2,3,figsize=(14,6)) # on a 6 cases

    for ax,img,title in zip(axes.ravel(),images,titles):
        if img.ndim==2: #image en niveau de gris
            ax.imshow(img,cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    
    axes.ravel()[-1].axis("off") # ne pas afficher la dernière case

    plt.tight_layout()  # pour ne pas que les titres se cascadent
    plt.show()
