# Filtres : Gaussien, Bilatéral , Médian et Non-Local Means

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

#les chemins et chargemnt de l'image
BASE_DIR = os.getcwd()
img_path = os.path.join(BASE_DIR, 'imdata', 'cameraman.tif')

# 1. Chargement et Prétraitement

def load_image(path):
    """
    charger une image en BGR
    """
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {path}")
    return img

def to_gray(img_bgr):
    """
    convertir l'image en niveau de gris
    """
    gray = cv.cvtColor(img_bgr,cv.COLOR_BGR2GRAY)
    return gray

def normalize(img):
    """normaliser l'image en float32 dans [0,1]
    """
    img = img.astype(np.float32) / 255.0
    return img

# 2. Ajout du bruit gaussien
def add_gaussian_noise(img,mean=0.0,sigma=0.05):
    """
    ajouter un bruit gaussien
    """
    noise = np.random.normal(mean,sigma,img.shape).astype(np.float32) #génération du bruit
    noisy = img + noise
    noisy = np.clip(noisy,0.0,1.0)
    return noisy

# 3. Filtre de debruitage

def apply_gaussian_blur(img,ksize=(5,5),sigma=1.0):
    """
    appliquer un flou gaussien
    """
    if img.dtype != np.uint8:
        img_uint8 = (img*255).astype(np.uint8)
    else:
        img_uint8 = img
    
    blurred = cv.GaussianBlur(img_uint8,ksize,sigmaX=sigma,sigmaY=sigma)
    return blurred

def apply_bilateral_filter(img,d=9,sigma_color=75,sigma_space=75):
    """
    appliquer un flou bilateral
    """
    if img.dtype != np.uint8:
        img_uint8 = (img*255).astype(np.uint8)
    else:
        img_uint8 = img
    
    bilateral = cv.bilateralFilter(img_uint8,d=d,sigmaColor=sigma_color,sigmaSpace=sigma_space)
    return bilateral

def apply_median_filter(img,ksize=5):
    """
    appliquer un filtre median
    """
    if img.dtype != np.uint8:
        img_uint8 = (img*255).astype(np.uint8)
    else:
        img_uint8 = img
    
    median = cv.medianBlur(img_uint8,ksize)
    return median

def apply_nl_means_denoising(img,h=10,template_window_size=7,search_window_size=21):
    """
    applique le debruitage NL_Means (version rapide OPENCV) sur une image en niveau de gris.
    img : uint8 ou float [0,1]
    h :  paramètre de filtrage (force de debruitage)
    template_window_size= taille du patch (impair)
    search_window_size = taille de la fenêtre de recherche(impair)"""
    if img.dtype != np.uint8:
        img_uint8 = (img*255).astype(np.uint8)
    else:
        img_uint8 = img
    
    img_denoised = cv.fastNlMeansDenoising(img_uint8,None,h=h,templateWindowSize=template_window_size,searchWindowSize=search_window_size)

    return img_denoised




# 4. Visualisation

def show_image(images,titles,cmap='gray'):
    """
    affiche grille 2 x N 
    """

    n = len(images)
    cols = (n+1) //2
    rows = 2

    plt.figure(figsize=(4*cols,8))

    for i, (img,title) in enumerate(zip(images,titles)):
        plt.subplot(rows,cols,i+1)
        plt.imshow(img,cmap=cmap)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ==================================Mon pipline ==================================

def main():
    # 1) chargement
    img_bgr = load_image(img_path)

    # 2) conversion en niveau de gris

    gray = to_gray(img_bgr)

    # 3) normaliser dans [0,1]

    img_gray_norm = normalize(img_bgr)

    # 4) ajout d'un bruit gaussien

    img_noise = add_gaussian_noise(img_gray_norm,mean=0.0,sigma=0.08)

    # 5) Application des filtres de debruitage

    gaussian_debruitage = apply_gaussian_blur(img_noise,ksize=(7,7),sigma=1.5)
    bilateral_debruitage = apply_bilateral_filter(img_noise,d=9,sigma_color=75,sigma_space=75)
    median_debruitage = apply_median_filter(img_noise,ksize=5)
    nlm_debruitage = apply_nl_means_denoising(img_noise,h=15,template_window_size=7,search_window_size=21)

    # 6. Conversion pour affichage

    original_uint8 = (img_gray_norm*255).astype(np.uint8)
    noisy_uint8 = (img_noise*255).astype(np.uint8)

    images = [original_uint8,noisy_uint8,gaussian_debruitage,bilateral_debruitage,median_debruitage,nlm_debruitage]
    titles = ["original","bruité","flou gaussien","filtre bilatéral","filtre médian","NL-MEANS"]

    show_image(images,titles,cmap='gray')

if __name__ == '__main__':
    main()