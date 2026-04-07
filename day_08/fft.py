#fft

from matplotlib.mlab import magnitude_spectrum
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from numpy.ma import mask_rowcols

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

# Domaine frequentiel (fft)
def compute_fft(img_gray_norm):
    """
calculer la FFT 2D et la version centrée (fftshift).
retourne le spectre complexe centré"""
    # FFT 2D
    f = np.fft.fft2(img_gray_norm)
    # decalage des bases au centre
    fshift = np.fft.fftshift(f)
    return fshift

def compute_magnitude_spectrum(fshift):
    """
    calcule le spectre de magnitude (pour visualisation)"""
    magnitude = np.abs(fshift)
    #on utilise log pour mieux voir les details
    magnitude_spectrum = np.log(1+magnitude)
    return magnitude_spectrum

# Filtres frequentiels (masques)

def make_low_pass_filter(shape,cutoff):
    """
    créer un filtre passe-bas circulaire.
    shape : (H,W)
    cutoff: rayon de coupure (en pixels)"""
    rows, cols = shape
    crow, ccol = rows //2, cols //2
    Y, X = np.ogrid[:rows,:cols]
    dist = np.sqrt((X-ccol)**2 + (Y-crow)**2)

    mask = np.zeros((rows,cols),np.float32)
    mask[dist <= cutoff] = 1.0
    return mask

def make_high_pass_filter(shape,cutoff):
    """creer un filtre passe-haut circulaire"""
    lp = make_low_pass_filter(shape,cutoff)
    hp = 1.0 - lp
    return hp

# Application des filtres en fréquences

def apply_frequency_filter(fshift,mask):
    """applique un filtre fréquentiel (mask) au spectre fshift"""
    # on suppose que le mask de meme taille que fshift
    fshift_filtered = fshift*mask
    return fshift_filtered

# Retour au domaine spatial (IFFT)

def computer_ifft(fshift_filtered):
    """calculer l'image spatiale à partir du spectre filtre"""
    # on remet les basses frequences dans les coins
    f_ishift = np.fft.ifftshift(fshift_filtered)
    # IFFT 2D
    img_back = np.fft.ifft2(f_ishift)
    # on prend la partie réelle
    img_back = np.real(img_back)
    # on sature dans [0,1]
    img_back = np.clip(img_back,0.0,1.0)
    #convertir en uint8 pour affichage
    img_back_uint8 = (img_back*255).astype(np.uint8)
    return img_back_uint8

# Visualisation

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

# Pipeline complet :

def main():
    # Chargement de l'image
    img_bgr = load_image(img_path)
    # niveau de gris
    img_gray = to_gray(img_bgr)
    # normalisation dans  [0,1]
    img_gray_norm = normalize(img_gray)
    # FFT 
    fshift = compute_fft(img_gray_norm)
    magnitude_spectrum=compute_magnitude_spectrum(fshift)
    # filtre frequentiel
    shape = img_gray_norm.shape
    lp_mask = make_low_pass_filter(shape,cutoff=30)
    hp_mask = make_high_pass_filter(shape,cutoff=30)
    # application filtte
    fshift_lp = apply_frequency_filter(fshift,lp_mask)
    fshift_hp = apply_frequency_filter(fshift,hp_mask)
    # retour spatial
    img_lp = computer_ifft(fshift_lp)
    img_hp = computer_ifft(fshift_hp)
    # preparation affichage
    original_uint8 = (img_gray_norm*255).astype(np.uint8)
    mag_uint8 = (magnitude_spectrum / magnitude_spectrum.max() *255).astype(np.uint8)

    images = [original_uint8,mag_uint8,img_lp,img_hp]
    titles = ["original","spectre magnitude","passe-bas(lissage)","passe-haut (details/contours)"]

    show_image(images,titles,cmap='gray')


if __name__=="__main__":
    main()