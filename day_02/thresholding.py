import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# seuillage simple (global)
# seuillage adaptatif (threshold adaptive [mean, gauss])
# Methode d'Otsu

# Chemin absolu vers la racine du projet, indépedant du terminal
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, 'imdata', 'rice.png')

img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img is not None, f'Image non trouvée : {img_path}'

# Affichage d'image 
"""
plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()
"""
# Application du seuillage simple 
seuil, max_val = 100,255

"""th,img_th1 = cv.threshold(img,seuil,max_val,cv.THRESH_BINARY)  # si <= seuil => 0 sinon 255
th,img_th2 = cv.threshold(img,seuil,max_val,cv.THRESH_BINARY_INV) #si <= seuil => 255 sinon 0

titles = ['original', 'seuillage binaire','seuillage binaire inv']
images = [img,img_th1,img_th2] # pour faciliter leur affichage

#Créer une figure
fig, ax = plt.subplots(1,3,figsize= (10,4)) #afficher les 3 côtes à côtes

for i in range(3):
    ax[i].imshow(images[i],cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].set_xticks([]) #pour ne pas avoir les valeurs sur l'axe des x
    ax[i].set_yticks([])

plt.show()
"""
# seuillage adaptatif (threshold adaptive Gaussian et Mean)
"""th,img_th1 = cv.threshold(img,seuil,max_val,cv.THRESH_BINARY)  # si <= seuil => 0 sinon 255 
img_th2 = cv.adaptiveThreshold(img,max_val,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3) # seuil en fonction la moyenne du voisinage
img_th3 = cv.adaptiveThreshold(img,max_val,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)

# Tester les 3 algos, seuillage global , seuillage adaptatif par la moyenne et par Gauss (somme cumulée des pixels des voisins)
titles = ['seuillage global', 'Threshold mean ','Threshold adaptive']
images = [img_th1,img_th2,img_th3] # pour faciliter leur affichage
#Créer une figure
fig, ax = plt.subplots(1,3,figsize= (10,4)) #afficher les 3 côtes à côtes

for i in range(3):
    ax[i].imshow(images[i],cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].set_xticks([]) #pour ne pas avoir les valeurs sur l'axe des x
    ax[i].set_yticks([])

plt.show()
"""

# Methode d'OTSU => pour trouver le seuil optimal à travers l'histogramme 
th1,img_th4 = cv.threshold(img,seuil,max_val,cv.THRESH_BINARY+cv.THRESH_OTSU) #OTSU renvoie le seuil automatiquement et après on fait le seuillage globale
th1,img_th5 = cv.threshold(img,seuil,max_val,cv.THRESH_BINARY) # moi je fixe un seuil en fonction duquel on fera le seuillage sur toute l'image
titles = ['OTSU', 'seuillage global']
images = [img_th4,img_th5] # pour faciliter leur affichage
#Créer une figure
fig, ax = plt.subplots(1,2,figsize= (10,4)) #afficher les 3 côtes à côtes

for i in range(2):
    ax[i].imshow(images[i],cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].set_xticks([]) #pour ne pas avoir les valeurs sur l'axe des x
    ax[i].set_yticks([])

plt.show()